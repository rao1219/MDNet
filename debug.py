import os, platform
from lib.vdbc.dataset_factory import VDBC


def debug_vdbc():
    dbtype = 'OTB'

    plf = platform.system()
    if plf == 'Windows':
        dbpath = os.path.join('D:\\', 'dataset', 'OTB')
        gtpath = dbpath
    else:
        gtpath = None
        dbpath = None

    if gtpath is not None and dbpath is not None:
        vdbc = VDBC(dbtype=dbtype, dbpath=dbpath, gtpath=gtpath, flush=True)
        image_list, gt_info, folder_map = vdbc.get_db()
        # vdbc.build_data(params=(0.2, 0.2, 0.05, 0.7, 0.5),
        #                 num=12)
        print image_list['David']
        print 'End.'


if __name__ == '__main__':
    debug_vdbc()

