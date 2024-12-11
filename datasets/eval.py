import pprint


def eval_corrupt_wrapper(model, fn_test_corrupt, args_test_corrupt, io=None):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    DGCNN_OA = {
        'clean': 0.926,
        'scale': 0.906,
        'jitter': 0.684,
        'rotate': 0.785,
        'dropout_global': 0.752,
        'dropout_local': 0.793,
        'add_global': 0.705,
        'add_local': 0.725
    }
    DGCNN_OA_OBJ = {
        'clean': 0.397,
        'scale': 0.370,
        'jitter': 0.217,
        'rotate': 0.332,
        'dropout_global': 0.273,
        'dropout_local': 0.261,
        'add_global': 0.213,
        'add_local': 0.244
    }
    DGCNN_OA_scan = {
        'clean': 0.858,
        'scale': 0.578,
        'jitter': 0.456,
        'rotate': 0.733,
        'dropout_global': 0.622,
        'dropout_local': 0.697,
        'add_global': 0.540,
        'add_local': 0.773
    }
    corruption_dataset = 'MN40'

    if corruption_dataset == 'lvis':
        OA = DGCNN_OA_OBJ
    elif corruption_dataset == 'MN40':
        # print('MN40')
        OA = DGCNN_OA
    elif corruption_dataset == 'scan':
        OA = DGCNN_OA_scan
    else:
        OA = None
    OA_clean = None
    perf_all = {'OA': [], 'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, model=model, **args_test_corrupt)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            if io is None:
                pprint.pprint(test_perf, width=200)
            else:
                io.cprint(test_perf)
            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break

        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)

        if corruption_type != 'clean':

            perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - OA[corruption_type])
            perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (OA['clean'] - OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        if io is None:
            pprint.pprint(perf_corrupt, width=200)
        else:
            io.cprint(perf_corrupt)
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    if io is None:
        pprint.pprint(perf_all, width=200)
    else:
        io.cprint(perf_all)
