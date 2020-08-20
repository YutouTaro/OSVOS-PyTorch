from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/content/drive/My Drive/datasets/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/content/OSVOS-PyTorch/models'

    @staticmethod
    def models_dir():
        return "/content/OSVOS-PyTorch/models"

