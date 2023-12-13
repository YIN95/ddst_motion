from scripts.args import parse_train_loco_opt
from LMG import LMG


def train(opt):
    model = LMG()
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_loco_opt()
    train(opt)
