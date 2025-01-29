
DEBUGGING_MODE = False

if DEBUGGING_MODE:
    # TODO:: Only a debugging thing, no need to commit this
    from debugging.things_comparator import compare_tensors
    from debugging.things_comparator import visualize_3d_blazepose_comparison
    from debugging.transform import best_transform
    from debugging.dynamic_plotter import DynamicPlotter
    from matplotlib import pyplot as plot

EXPECT_WEBCAM_ONLY = False

def dprint(*args, **kwargs):
    if DEBUGGING_MODE:
        print(*args, **kwargs)

def map_to_range(input_value, n, m):
    """Maps input_value from [0, n-1] to [0, m-1] using rounding."""
    return int(round((input_value / (n - 1)) * (m - 1)))  #Scale 

