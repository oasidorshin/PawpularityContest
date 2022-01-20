import os

from data import ImgDataset, get_train_transform, get_val_transform, validation
from modelling import BaseTransformer, BCE_scaled, MSE_scaled
from training import train_loop
from utils import seed_everything, get_timestamp
import logging, argparse

TELEGRAM_SEND = True


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Data params
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--n_bins', default=20, type=int)  # for validation

    # Model params
    parser.add_argument('--model_name', default="swin_large_patch4_window7_224", type=str)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--n_classes', default=1, type=int)

    # Training
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    parser.add_argument('--lr', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--warmup_coef', default=0.1, type=float)  # for scheduler
    parser.add_argument('--clip_grad_norm', default=0, type=float)

    # Augmentation
    parser.add_argument('--cj_intensity', default=0, type=float)
    parser.add_argument('--cj_p', default=0, type=float)

    parser.add_argument('--mixup', default=0.1, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    # Get foldername
    foldername = get_timestamp()
    os.mkdir(f"models/{foldername}")

    # Set up logging
    logging.basicConfig(filename=f'models/{foldername}/training.log', encoding='utf-8', level=logging.DEBUG)
    logging.info(f"STARTING TIME {foldername}")

    # Parse arguments
    args = parse_args()
    seed_everything(args.seed)
    logging.info(f"ARGUMENTS {args} \n")

    # Begin training
    train_dataset = ImgDataset(transform=get_train_transform(args.img_size, args.cj_intensity, args.cj_p),
                               folder="train")
    val_dataset = ImgDataset(transform=get_val_transform(args.img_size), folder="train")

    train_loop(BaseTransformer,
               BCE_scaled(),
               MSE_scaled(),
               train_dataset,
               val_dataset,
               validation(args.n_bins, args.n_splits, args.seed),
               [0, 1],
               foldername,
               args)

    logging.info(f"DONE AT {get_timestamp()}")

    if TELEGRAM_SEND:
        import telegram_send

        telegram_send.send(messages=[f"Script {__file__} is done"])
