from numba import njit, prange
from numpy import empty as np_empty, float32 as np_float32, sum as np_sum

# constraints computation is the main bottleneck of CAN: make sure the
# functions defined in this file are as fast as possible. Compiling them can
# lead to significant improvements. Numba is a nice library for JIT compilation
# and, as an additional bonus, it also provides a mechanism to disable Python's
# GIL and enable real multi-threading. Only a strict subset of Numpy methods
# and attributes are supported by Numba: pay special attention! Including the
# entire Numpy library could be error-prone: import only what is strictly
# required.
# ref: https://numba.pydata.org/numba-doc/dev/user/jit.html
# ref: https://numba.pydata.org/numba-doc/latest/user/parallel.html
# ref: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

# each constraint must be a real value in [0, 1]. 0: perfect; 1: totally wrong.


@njit(nogil=True, parallel=True)
def area_and_convexity_single_batch(
        data_mb, num_constraints=None, area=None, polygons_number=None,
        img_width=None, img_height=None, **kwargs):
    # preallocate an output array that will contain the computed
    # constraints values for the batch
    mb_constraints_values = np_empty(
        shape=(data_mb.shape[0], num_constraints), dtype=np_float32)

    for i in prange(data_mb.shape[0]):
        sample = data_mb[i]
        # preallocate an output array that will contain the computed
        # constraints value for the i-th element
        constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

        target_area = area * polygons_number
        nonzero = np_sum(sample)
        norm = img_width * img_height - target_area
        greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
        smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
        constraints[0] = greater_area_inner
        constraints[1] = smaller_area_inner
        # convexity
        constraints[2] = _convex(sample, img_width, img_height)
        mb_constraints_values[i] = constraints
    return mb_constraints_values


@njit(nogil=True, parallel=True)
def area_and_convexity_multi_batch(
        R_mb, n_samples=None, bs=None, num_constraints=None, area=None,
        polygons_number=None, img_width=None, img_height=None, **kwargs):
    # we have (n_samples x bs) generated items; for each of them we have to
    # compute the values of each constraint.
    # R_mb has shape [n_samples, bs] + image_shape
    # e.g. [n_samples, bs, 20, 20, 1]
    # so we need to iterate over the first two dimensions

    # preallocate an output array that will contain the computed
    # constraints values for the group of batches
    R_mb_constraints_values = np_empty(
        shape=(n_samples, bs, num_constraints), dtype=np_float32)

    for i in prange(R_mb.shape[0]):
        for j in prange(R_mb.shape[1]):
            sample = R_mb[i][j]
            # preallocate an output array that will contain the computed
            # constraints value for the i-th element
            constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

            target_area = area * polygons_number
            nonzero = np_sum(sample)
            norm = img_width * img_height - target_area
            greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
            smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
            constraints[0] = greater_area_inner
            constraints[1] = smaller_area_inner
            # convexity
            constraints[2] = _convex(sample, img_width, img_height)
            R_mb_constraints_values[i][j] = constraints
    return R_mb_constraints_values


@njit(nogil=True, parallel=True)
def area_and_parity_check_single_batch(
        data_mb, num_constraints=None, area=None, polygons_number=None,
        img_width=None, img_height=None, **kwargs):
    # preallocate an output array that will contain the computed
    # constraints values for the batch
    mb_constraints_values = np_empty(
        shape=(data_mb.shape[0], num_constraints), dtype=np_float32)

    for i in prange(data_mb.shape[0]):
        sample = data_mb[i]
        # preallocate an output array that will contain the computed
        # constraints value for the i-th element
        constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

        target_area = area * polygons_number
        nonzero = np_sum(sample[1:-1, 1:-1])
        norm = img_width * img_height - target_area
        greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
        smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
        constraints[0] = greater_area_inner
        constraints[1] = smaller_area_inner
        for x in prange(1, 10):  # _parity_check_rl_X
            sum_left = np_sum(sample[x, 1:int(img_width / 2)])
            constraints[x + 1] = int(sample[x][0][0] != sum_left % 2)
        for x in prange(1, 10):  # _parity_check_ct_X
            sum_top = np_sum(sample[1:int(img_height / 2), x])
            constraints[x + 10] = int(sample[0][x][0] != sum_top % 2)
        mb_constraints_values[i] = constraints
    return mb_constraints_values


@njit(nogil=True, parallel=True)
def area_and_parity_check_multi_batch(
        R_mb, n_samples=None, bs=None, num_constraints=None, area=None,
        polygons_number=None, img_width=None, img_height=None, **kwargs):
    # we have (n_samples x bs) generated items; for each of them we have to
    # compute the values of each constraint.
    # R_mb has shape [n_samples, bs] + image_shape
    # e.g. [n_samples, bs, 20, 20, 1]
    # so we need to iterate over the first two dimensions

    # preallocate an output array that will contain the computed
    # constraints values for the group of batches
    R_mb_constraints_values = np_empty(
        shape=(n_samples, bs, num_constraints), dtype=np_float32)

    for i in prange(R_mb.shape[0]):
        for j in prange(R_mb.shape[1]):
            sample = R_mb[i][j]
            # preallocate an output array that will contain the computed
            # constraints value for the i-th element
            constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

            target_area = area * polygons_number
            nonzero = np_sum(sample[1:-1, 1:-1])
            norm = img_width * img_height - target_area
            greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
            smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
            constraints[0] = greater_area_inner
            constraints[1] = smaller_area_inner
            for x in prange(1, 10):  # _parity_check_rl_X
                sum_left = np_sum(sample[x, 1:int(img_width / 2)])
                constraints[x + 1] = int(sample[x][0][0] != sum_left % 2)
            for x in prange(1, 10):  # _parity_check_ct_X
                sum_top = np_sum(sample[1:int(img_height / 2), x])
                constraints[x + 10] = int(sample[0][x][0] != sum_top % 2)
            R_mb_constraints_values[i][j] = constraints
    return R_mb_constraints_values


@njit(nogil=True, parallel=True)
def area_and_all_parity_check_single_batch(
        data_mb, num_constraints=None, area=None, polygons_number=None,
        img_width=None, img_height=None, **kwargs):
    # preallocate an output array that will contain the computed
    # constraints values for the batch
    mb_constraints_values = np_empty(
        shape=(data_mb.shape[0], num_constraints), dtype=np_float32)

    for i in prange(data_mb.shape[0]):
        sample = data_mb[i]
        # preallocate an output array that will contain the computed
        # constraints value for the i-th element
        constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

        target_area = area * polygons_number
        nonzero = np_sum(sample[1:-1, 1:-1])
        norm = img_width * img_height - target_area
        greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
        smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
        constraints[0] = greater_area_inner
        constraints[1] = smaller_area_inner
        curr_index = 1
        for x in prange(1, img_width - 1):  # _parity_check_rl_X
            sum_left = np_sum(sample[x, 1:int(img_width / 2)])
            constraints[x + curr_index] = int(sample[x][0][0] != sum_left % 2)
        curr_index += img_width - 2
        for x in prange(1, img_height - 1):  # _parity_check_ct_X
            sum_top = np_sum(sample[1:int(img_height / 2), x])
            constraints[x + curr_index] = int(sample[0][x][0] != sum_top % 2)
        curr_index += img_height - 2
        for x in prange(1, img_width - 1):  # _parity_check_rr_X
            sum_right = np_sum(sample[x, int(img_width / 2):img_width - 1])
            constraints[x + curr_index] = int(
                sample[x][img_width - 1][0] != sum_right % 2)
        curr_index += img_width - 2
        for x in prange(1, img_height - 1):  # _parity_check_cb_X
            sum_bottom = np_sum(sample[int(img_height / 2):img_height - 1, x])
            constraints[x + curr_index] = int(
                sample[img_height - 1][x][0] != sum_bottom % 2)
        mb_constraints_values[i] = constraints
    return mb_constraints_values


@njit(nogil=True, parallel=True)
def area_and_all_parity_check_multi_batch(
        R_mb, n_samples=None, bs=None, num_constraints=None, area=None,
        polygons_number=None, img_width=None, img_height=None, **kwargs):
    # we have (n_samples x bs) generated items; for each of them we have to
    # compute the values of each constraint.
    # R_mb has shape [n_samples, bs] + image_shape
    # e.g. [n_samples, bs, 20, 20, 1]
    # so we need to iterate over the first two dimensions

    # preallocate an output array that will contain the computed
    # constraints values for the group of batches
    R_mb_constraints_values = np_empty(
        shape=(n_samples, bs, num_constraints), dtype=np_float32)

    for i in prange(R_mb.shape[0]):
        for j in prange(R_mb.shape[1]):
            sample = R_mb[i][j]
            # preallocate an output array that will contain the computed
            # constraints value for the i-th element
            constraints = np_empty(shape=(num_constraints,), dtype=np_float32)

            target_area = area * polygons_number
            nonzero = np_sum(sample[1:-1, 1:-1])
            norm = img_width * img_height - target_area
            greater_area_inner = min(1, max(0, nonzero - target_area) / norm)
            smaller_area_inner = min(1, max(0, target_area - nonzero) / norm)
            constraints[0] = greater_area_inner
            constraints[1] = smaller_area_inner
            curr_index = 1
            for x in prange(1, img_width - 1):  # _parity_check_rl_X
                sum_left = np_sum(sample[x, 1:int(img_width / 2)])
                constraints[x + curr_index] = int(
                    sample[x][0][0] != sum_left % 2)
            curr_index += img_width - 2
            for x in prange(1, img_height - 1):  # _parity_check_ct_X
                sum_top = np_sum(sample[1:int(img_height / 2), x])
                constraints[x + curr_index] = int(
                    sample[0][x][0] != sum_top % 2)
            curr_index += img_height - 2
            for x in prange(1, img_width - 1):  # _parity_check_rr_X
                sum_right = np_sum(sample[x, int(img_width / 2):img_width - 1])
                constraints[x + curr_index] = int(
                    sample[x][img_width - 1][0] != sum_right % 2)
            curr_index += img_width - 2
            for x in prange(1, img_height - 1):  # _parity_check_cb_X
                sum_bottom = np_sum(
                    sample[int(img_height / 2):img_height - 1, x])
                constraints[x + curr_index] = int(
                    sample[img_height - 1][x][0] != sum_bottom % 2)
            R_mb_constraints_values[i][j] = constraints
    return R_mb_constraints_values


@njit(nogil=True, parallel=True)
def _convex(sample, img_width, img_height):
    # sample shape [20, 20, 1]
    sample = sample.reshape(sample.shape[:-1])

    visited_pixel = {(0, 0)}  # trick to let Numba infer set type
    visited_pixel.clear()  # remove placeholder element
    error = 0
    for w in range(img_width):
        for h in range(img_height):
            c = (w, h)
            if c in visited_pixel or sample[c] != 1:
                continue
            scc = {c}  # trick to let Numba infer set type
            _bfs(sample, visited_pixel, c, scc, img_width, img_height)

            # compute SCC error

            # partition by row
            for i in prange(img_height):
                error += _axis_error([p for p in scc if p[1] == i], 0)

            # partition by col
            for i in prange(img_width):
                error += _axis_error([p for p in scc if p[0] == i], 1)

            # partition by diag_R
            for i in prange(img_width + img_height):
                error += _axis_error([p for p in scc if p[0] + p[1] == i], 0)

            # partition by diag_L
            for i in prange(-img_height, img_width):
                error += _axis_error([p for p in scc if p[1] - p[0] == i], 0)
    return min(1, error / (2 * (img_width * img_height)))


@njit(nogil=True, parallel=True)
def _bfs(sample, visited_pixel, c, scc, img_width, img_height):
    scc.add(c)
    visited_pixel.add(c)
    x, y = c

    up_value = c[0] - 1, c[1]
    if x > 0 and sample[up_value] == 1 \
            and up_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_value, scc, img_width, img_height)

    up_l_value = c[0] - 1, c[1] - 1
    if x > 0 and y > 0 and sample[up_l_value] == 1 \
            and up_l_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_l_value, scc, img_width, img_height)

    up_r_value = c[0] - 1, c[1] + 1
    if x > 0 and y < img_width - 2 and sample[up_r_value] == 1 \
            and up_r_value not in visited_pixel:
        _bfs(sample, visited_pixel, up_r_value, scc, img_width, img_height)

    l_value = c[0], c[1] - 1
    if y > 0 and sample[l_value] == 1 \
            and l_value not in visited_pixel:
        _bfs(sample, visited_pixel, l_value, scc, img_width, img_height)

    r_value = c[0], c[1] + 1
    if y < img_width - 2 and sample[r_value] == 1 \
            and r_value not in visited_pixel:
        _bfs(sample, visited_pixel, r_value, scc, img_width, img_height)

    down_value = c[0] + 1, c[1]
    if x < img_height - 2 and sample[down_value] == 1 \
            and down_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_value, scc, img_width, img_height)

    down_l_value = c[0] + 1, c[1] - 1
    if x < img_height - 2 and y > 0 and sample[down_l_value] == 1 \
            and down_l_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_l_value, scc, img_width, img_height)

    down_r_value = c[0] + 1, c[1] + 1
    if x < img_height - 2 and y < img_width - 2 and sample[down_r_value] == 1 \
            and down_r_value not in visited_pixel:
        _bfs(sample, visited_pixel, down_r_value, scc, img_width, img_height)


@njit(nogil=True, parallel=True)
def _min_max(items):
    curr_min = curr_max = items[0]
    for i in items:
        if i < curr_min:
            curr_min = i
        if i > curr_max:
            curr_max = i
    return curr_min, curr_max


@njit(nogil=True)
def _axis_error(iterable, axis):
    items = [i[axis] for i in iterable]
    if len(items) == 0:
        return 0
    min_item, max_item = _min_max(items)
    return max_item - min_item - len(items) + 1
