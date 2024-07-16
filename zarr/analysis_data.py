import numpy as np
import zarr

def feature_test():
    # Make test sets to verify the function
    dataset_a1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_a2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a2-c.zarr'), [f"feature{i}" for i in range(10)])
    }
    dataset_a3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 10)), store='feature/a3-d.zarr'), [f"feature{i}" for i in range(10)])
    }

    test_set1 = [dataset_a1, dataset_a2, dataset_a3]

    # Larger test set using store and 100 rows and 100 columns
    dataset_b1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b1-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b1-b.zarr'), [f"feature{i}" for i in range(100)])
        }
    dataset_b2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b2-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b2-c.zarr'), [f"feature{i}" for i in range(100)])
    }
    dataset_b3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b3-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 100)), store='feature/b3-b.zarr'), [f"feature{i}" for i in range(100)])
    }
    test_set2 = [dataset_b1, dataset_b2, dataset_b3]

    dataset_c1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c1-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c1-b.zarr'), [f"feature{i}" for i in range(1000)])
        }
    dataset_c2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c2-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c2-c.zarr'), [f"feature{i}" for i in range(1000)])
    }
    dataset_c3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c3-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 1000)), store='feature/c3-d.zarr'), [f"feature{i}" for i in range(1000)])
    }
    test_set3 = [dataset_c1, dataset_c2, dataset_c3]

    dataset_d1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d1-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d1-b.zarr'), [f"feature{i}" for i in range(10000)])
        }
    dataset_d2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d2-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d2-c.zarr'), [f"feature{i}" for i in range(10000)])
    }
    dataset_d3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d3-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 10000)), store='feature/d3-d.zarr'), [f"feature{i}" for i in range(10000)])
    }
    test_set4 = [dataset_d1, dataset_d2, dataset_d3]

    dataset_e1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e1-a.zarr'), [f"feature{i}" for i in range(100000)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e1-b.zarr'), [f"feature{i}" for i in range(100000)])
        }
    dataset_e2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e2-a.zarr'), [f"feature{i}" for i in range(100000)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e2-c.zarr'), [f"feature{i}" for i in range(100000)])
    }
    dataset_e3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e3-a.zarr'), [f"feature{i}" for i in range(100000)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 100000)), store='feature/e3-d.zarr'), [f"feature{i}" for i in range(100000)])
    }
    test_set5 = [dataset_e1, dataset_e2, dataset_e3]

    return test_set1, test_set2, test_set3, test_set4, test_set5

def row_test(chunk_size:int=2500):
    # Make test sets to verify the function
    dataset_a1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_a2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a2-c.zarr'), [f"feature{i}" for i in range(10)]),
    }
    dataset_a3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (chunk_size, 10), store='row/a3-d.zarr'), [f"feature{i}" for i in range(10)])
    }

    test_set1 = [dataset_a1, dataset_a2, dataset_a3]

    # Larger test set using store and 100 rows and 100 columns
    dataset_b1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_b2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b2-c.zarr'), [f"feature{i}" for i in range(10)])
    }
    dataset_b3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (100, 10)), chunk = (chunk_size, 10), store='row/b3-d.zarr'), [f"feature{i}" for i in range(10)])
    }
    test_set2 = [dataset_b1, dataset_b2, dataset_b3]

    dataset_c1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_c2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c2-c.zarr'), [f"feature{i}" for i in range(10)])
    }
    dataset_c3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (1000, 10)), chunk = (chunk_size, 10), store='row/c3-d.zarr'), [f"feature{i}" for i in range(10)])
    }
    test_set3 = [dataset_c1, dataset_c2, dataset_c3]

    dataset_d1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_d2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d2-c.zarr'), [f"feature{i}" for i in range(10)])
    }
    dataset_d3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (10000, 10)), chunk = (chunk_size, 10), store='row/d3-d.zarr'), [f"feature{i}" for i in range(10)])
    }
    test_set4 = [dataset_d1, dataset_d2, dataset_d3]

    dataset_e1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_e2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e2-c.zarr'), [f"feature{i}" for i in range(10)])
    }
    dataset_e3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (100000, 10)), chunk = (chunk_size, 10), store='row/e3-d.zarr'), [f"feature{i}" for i in range(10)])
    }
    test_set5 = [dataset_e1, dataset_e2, dataset_e3]
    
    # Could Not test because of memory error
    # dataset_f1 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f1-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_B": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f1-b.zarr'), [f"feature{i}" for i in range(10)])
    #     }
    # dataset_f2 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f2-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_C": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f2-c.zarr'), [f"feature{i}" for i in range(10)]),
    # }
    # dataset_f3 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f3-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_D": (zarr.array(np.random.randint(0, 100, (1000000, 10)), chunk = (chunk_size, 10), store='row/f3-d.zarr'), [f"feature{i}" for i in range(10)])
    # }
    # test_set6 = [dataset_f1, dataset_f2, dataset_f3]

    # dataset_g1 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g1-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_B": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g1-b.zarr'), [f"feature{i}" for i in range(10])]
    #     }
    # dataset_g2 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g2-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_C": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g2-c.zarr'), [f"feature{i}" for i in range(10)]),
    # }
    # dataset_g3 = {
    #     "type_A": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g3-a.zarr'), [f"feature{i}" for i in range(10)]),
    #     "type_D": (zarr.array(np.random.randint(0, 100, (10000000, 10)), chunk = (chunk_size, 10), store='row/g3-d.zarr'), [f"feature{i}" for i in range(10)])
    # }
    # test_set7 = [dataset_g1, dataset_g2, dataset_g3]
    return test_set1, test_set2, test_set3, test_set4, test_set5

def mix_test(chunk_size:int=2500):
    dataset_a1 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a1-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_B": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a1-b.zarr'), [f"feature{i}" for i in range(10)])
        }
    dataset_a2 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a2-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_C": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a2-c.zarr'), [f"feature{i}" for i in range(10)]),
    }
    dataset_a3 = {
        "type_A": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a3-a.zarr'), [f"feature{i}" for i in range(10)]),
        "type_D": (zarr.array(np.random.randint(0, 10, (10, 10)), chunk = (2500, 10), store='mix/a3-d.zarr'), [f"feature{i}" for i in range(10)])
    }

    test_set1 = [dataset_a1, dataset_a2, dataset_a3]

    dataset_b1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b1-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b1-b.zarr'), [f"feature{i}" for i in range(100)])
    }
    dataset_b2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b2-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b2-c.zarr'), [f"feature{i}" for i in range(100)]
        )
    }
    dataset_b3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b3-a.zarr'), [f"feature{i}" for i in range(100)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (100, 100)), chunk = (2500, 100), store='mix/b3-d.zarr'), [f"feature{i}" for i in range(100)]
        )
    }
    test_set2 = [dataset_b1, dataset_b2, dataset_b3]

    dataset_c1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c1-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c1-b.zarr'), [f"feature{i}" for i in range(1000)])
    }
    dataset_c2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c2-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c2-c.zarr'), [f"feature{i}" for i in range(1000)]
        )
    }
    dataset_c3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c3-a.zarr'), [f"feature{i}" for i in range(1000)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (1000, 1000)), chunk = (2500, 1000), store='mix/c3-d.zarr'), [f"feature{i}" for i in range(1000)]
        )
    }
    test_set3 = [dataset_c1, dataset_c2, dataset_c3]

    dataset_d1 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d1-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_B": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d1-b.zarr'), [f"feature{i}" for i in range(10000)]
        )
    }

    dataset_d2 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d2-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_C": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d2-c.zarr'), [f"feature{i}" for i in range(10000)]
        )
    }
    dataset_d3 = {
        "type_A": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d3-a.zarr'), [f"feature{i}" for i in range(10000)]),
        "type_D": (zarr.array(np.random.randint(0, 100, (10000, 10000)), chunk = (2500, 10000), store='mix/d3-d.zarr'), [f"feature{i}" for i in range(10000)]
        )
    }
    test_set4 = [dataset_d1, dataset_d2, dataset_d3]
    return test_set1, test_set2, test_set3, test_set4