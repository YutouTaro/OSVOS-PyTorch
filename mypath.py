from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/content/drive/My Drive/datasets/PASCAL2010'
        # return '/home/yu/DataSet/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/content/drive/My Drive/datasets/PASCAL2010/models'
        # return '/home/yu/OSVOS-PyTorch/models'

    @staticmethod
    def models_dir():
        return "/content/drive/My Drive/datasets/PASCAL2010/models"
        # return '/home/yu/OSVOS-PyTorch/models'