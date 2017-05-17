import os, platform
from lib.vdbc.dataset_factory import VDBC

EXCLUDE_SET = {
    'vot2014': ['Basketball', 'Bolt', 'David', 'Diving',
                'MotorRolling', 'Skating1', 'Trellis', 'Woman']}


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
        vdbc.del_exclude(EXCLUDE_SET['vot2014'])
        image_list, gt_info, folder_map = vdbc.get_db()
        datas = vdbc.build_data_in_list_order(params=(0.2, 0.2, 0.05, 0.7, 0.5),
                        num=12)

        print 'End.'


if __name__ == '__main__':
    debug_vdbc()

