import argparse
import logging
import time

import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results
from utils.model_util import create_diffusion
import inference.models.lgdm.albef.utils as utils

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.01,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true',
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anywhere',
                        help='Specific for Grasp-Anywhere')
    
    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args




def _init_process():
    class WrapperArgument:
            def __init__(self):
                pass

            def add_attribute(self, name, value):
                setattr(self, name, value)

    args = WrapperArgument()
    args.config = 'inference/models/lgdm/albef/configs/Grounding.yaml'
    args.gradcam_mode = 'itm'
    args.block_num = 8
    args.text_encoder = 'bert-base-uncased'
    args.device = 'cuda'
    args.world_size = 1
    args.dist_url = 'env://'
    args.distributed = True

    utils.init_distributed_mode(args)

if __name__ == '__main__':
    _init_process()
    args = parse_args()
    # Set seed for CPU
    torch.manual_seed(args.random_seed)

    # Set seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # Get the compute device
    device = get_device(args.force_cpu)
    diffusion = create_diffusion()

    sample_fn = (
            diffusion.p_sample_loop
    )

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb,
                           seen=args.seen,
                           add_file_path=args.add_file_path)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    # test_dataset = sorted(test_dataset, key=lambda x: x[5])

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        # Load Network
        net = torch.load(network)

        results = {'correct': 0, 'failed': 0}

        if args.jacquard_output:
            jo_fn = network + '_jacquard_output.txt'
            with open(jo_fn, 'w') as f:
                pass

        start_time = time.time()

        with torch.no_grad():
            for idx, (x, y, didx, rot, zoom, prompt, query) in enumerate(test_data):
                img = x.to(device)
                yc = [yi.to(device) for yi in y]
                pos_gt = yc[0]

                alpha = 0.4
                idx = torch.zeros(img.shape[0]).to(device)

                sample = sample_fn(
                    net,
                    pos_gt.shape,
                    pos_gt,
                    img,
                    query,
                    alpha,
                    idx,
                )

                pos_output = sample
                cos_output, sin_output, width_output = net.cos_output_str, net.sin_output_str, net.width_output_str

                # pos_output, cos_output, sin_output, width_output = net(None, img, None, query, alpha, idx)
                lossd = net.compute_loss(yc, pos_output, cos_output, sin_output, width_output)

                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                       no_grasps=args.n_grasps,
                                                       grasp_width=width_img,
                                                       threshold=args.iou_threshold
                                                       )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    with open(jo_fn, 'a') as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + '\n')
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                if args.vis:
                    save_results(
                        rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=args.n_grasps,
                        grasp_width_img=width_img
                    )

        avg_time = (time.time() - start_time) / len(test_data)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        if args.jacquard_output:
            logging.info('Jacquard output saved to {}'.format(jo_fn))

        del net
        torch.cuda.empty_cache()
