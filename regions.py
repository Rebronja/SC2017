import skimage.measure
import hirautil
import operator


def extract(img):
    labeled_img = skimage.measure.label(1 - img)
    regions = skimage.measure.regionprops(labeled_img)

    mx = region_max(regions)
    noise = 10

    rgs = []
    for region in regions:
        w = region.bbox[3] - region.bbox[1]
        h = region.bbox[2] - region.bbox[0]
        if w > mx / noise and h > mx / noise:
            rgs.append(hirautil.Region(region))
    rgs = perform_overlap(rgs, mx)
    rgs = perform_overlap(rgs, mx)
    rgs = perform_up(rgs, mx)
    rgs = perform_overlap(rgs, mx)
    rgs = perform_right(rgs, mx)
    rgs = perform_overlap(rgs, mx)
    rgs = perform_handakuten(rgs, mx)
    rgs = perform_overlap(rgs, mx)
    rgs = perform_handakuten(rgs, mx)
    rgs = perform_overlap(rgs, mx)

    return rgs


def perform_up(rgs, mx):
    rgs = sorted(rgs, key=operator.attrgetter('x'))

    for reg in rgs:
        neighbourhood(reg, rgs, mx)

    tmp = []
    for idx in range(len(rgs)):
        reg = rgs[idx]
        if not reg.used:
            for jdx in range(len(reg.cands)):
                cand = rgs[idx].cands[jdx]
                if not rgs[cand].used:
                    if check_up(reg, rgs[cand]) or check_up_ko(reg, rgs[cand], mx):
                        rgs[cand].used = True
                        rgs[idx].used = True
                        new = hirautil.Region()
                        new.bbox = merge_up(reg, rgs[cand])
                        new.x = new.bbox[1]
                        tmp.append(new)
                        break
            if not rgs[idx].used:
                tmp.append(rgs[idx])

    rgs = sorted(tmp, key=operator.attrgetter('x'))
    return rgs


def perform_overlap(rgs, mx):
    tmp = []

    for reg in rgs:
        neighbourhood(reg, rgs, mx)

    for idx in range(len(rgs)):
        reg = rgs[idx]
        if not reg.used:
            for jdx in range(len(reg.cands)):
                cand = rgs[idx].cands[jdx]
                if not rgs[cand].used:
                    if check_overlap(reg, rgs[cand]):
                        rgs[cand].used = True
                        rgs[idx].used = True
                        new = hirautil.Region()
                        new.bbox = merge_up(reg, rgs[cand])
                        new.x = new.bbox[1]
                        tmp.append(new)
                        break
            if not rgs[idx].used:
                tmp.append(rgs[idx])

    rgs = sorted(tmp, key=operator.attrgetter('x'))
    return rgs


def perform_right(rgs, mx):
    tmp = []

    for reg in rgs:
        neighbourhood(reg, rgs, mx)

    for idx in range(len(rgs)):
        reg = rgs[idx]
        if not reg.used:
            for jdx in range(len(reg.cands)):
                cand = rgs[idx].cands[jdx]
                if not rgs[cand].used:
                    if check_right(reg, rgs[cand], mx):
                        rgs[cand].used = True
                        rgs[idx].used = True
                        new = hirautil.Region()
                        new.bbox = merge_up(reg, rgs[cand])
                        new.x = new.bbox[1]
                        tmp.append(new)
                        break
            if not rgs[idx].used:
                tmp.append(rgs[idx])

    rgs = sorted(tmp, key=operator.attrgetter('x'))
    return rgs


def perform_handakuten(rgs, mx):
    tmp = []

    for reg in rgs:
        neighbourhood(reg, rgs, mx)

    for idx in range(len(rgs)):
        reg = rgs[idx]
        if not reg.used:
            for jdx in range(len(reg.cands)):
                cand = rgs[idx].cands[jdx]
                if not rgs[cand].used:
                    if check_handakuten(reg, rgs[cand], mx):
                        rgs[cand].used = True
                        rgs[idx].used = True
                        new = hirautil.Region()
                        new.bbox = merge_up(reg, rgs[cand])
                        new.x = new.bbox[1]
                        tmp.append(new)
                        break
            if not rgs[idx].used:
                tmp.append(rgs[idx])

    rgs = sorted(tmp, key=operator.attrgetter('x'))
    return rgs


def region_max(regions):
    widths = []
    heights = []
    for region in regions:
        widths.append(region.bbox[3] - region.bbox[1])
        heights.append(region.bbox[2] - region.bbox[0])

    xmax = max(widths)
    ymax = max(heights)
    return max(xmax, ymax)


def dist(first, second, reach):
    s_y1, s_y2, s_x1, s_x2 = second.bbox[0], second.bbox[2], second.bbox[1], second.bbox[3]
    f_y1, f_y2, f_x1, f_x2 = first.bbox[0], first.bbox[2], first.bbox[1], first.bbox[3]
    if f_y1 - reach <= s_y1 <= f_y2 + reach and f_x1 - reach <= s_x1 <= f_x2 + reach:
        return True
    if f_y1 - reach <= s_y2 <= f_y2 + reach and f_x1 - reach <= s_x2 <= f_x2 + reach:
        return True
    if s_y1 - reach <= f_y1 <= s_y2 + reach and s_x1 - reach <= f_x1 <= s_x2 + reach:
        return True
    if s_y1 - reach <= f_y2 <= s_y2 + reach and s_x1 - reach <= f_x2 <= s_x2 + reach:
        return True

    return False


def neighbourhood(current, regions, reach):
    current.cands = []
    for idx in range(len(regions)):
        if current.bbox != regions[idx].bbox:
            if dist(current, regions[idx], reach):
                current.cands.append(idx)


def check_up(first, second):
    fr_h = first.bbox[2] - first.bbox[0]
    fr_w = first.bbox[3] - first.bbox[1]
    sc_w = second.bbox[3] - second.bbox[1]
    if second.bbox[0] < first.bbox[0] and second.bbox[2] <= first.bbox[0] + fr_h / 5:
        if second.bbox[0] > first.bbox[0] - fr_h / 2:
            if sc_w <= fr_w * 3/2:
                return True

    return False


def check_up_ko(first, second, mx):
    fr_w = first.bbox[3] - first.bbox[1]
    sc_w = second.bbox[3] - second.bbox[1]
    fr_h = first.bbox[2] - first.bbox[0]
    sc_h = second.bbox[2] - second.bbox[0]
    if fr_w > fr_h and sc_w > sc_h:
        if fr_h * 3/2 < fr_w:
            if second.bbox[0] < first.bbox[0] and second.bbox[2] <= first.bbox[0]:
                # if second.bbox[0] + sc_h * 2/3 > first.bbox[0] - fr_h * 5/2:
                if second.bbox[0] > first.bbox[2] - mx:
                    if second.bbox[1] >= first.bbox[1] and second.bbox[3] <= first.bbox[3]:
                        return True

    return False


def check_overlap(first, second):
    if first.bbox[0] <= second.bbox[0] <= first.bbox[2] and first.bbox[1] <= second.bbox[1] <= first.bbox[3]:
        return True
    if first.bbox[0] <= second.bbox[2] <= first.bbox[2] and first.bbox[1] <= second.bbox[3] <= first.bbox[3]:
        return True
    if second.bbox[0] <= first.bbox[0] <= second.bbox[2] and second.bbox[1] <= first.bbox[1] <= second.bbox[3]:
        return True
    if second.bbox[0] <= first.bbox[2] <= second.bbox[2] and second.bbox[1] <= first.bbox[3] <= second.bbox[3]:
        return True

    return False


def check_right(first, second, mx):
    fr_w = first.bbox[3] - first.bbox[1]
    sc_w = second.bbox[3] - second.bbox[1]
    fr_h = first.bbox[2] - first.bbox[0]
    sc_h = second.bbox[2] - second.bbox[0]
    if second.bbox[1] > first.bbox[3] and second.bbox[1] < first.bbox[1] + mx:
        if fr_h > fr_w * 2:
            if first.bbox[1] + fr_w * 2 >= second.bbox[1]:
                return True
        if fr_h > sc_h and fr_w > sc_w:
            if first.bbox[1] + fr_w * 2 >= second.bbox[1]:
                return True

    return False


def check_handakuten(first, second, mx, ratio=3):
    fr_w = first.bbox[3] - first.bbox[1]
    sc_w = second.bbox[3] - second.bbox[1]
    fr_h = first.bbox[2] - first.bbox[0]
    sc_h = second.bbox[2] - second.bbox[0]
    if mx > sc_w * 4:
        if first.bbox[0] - fr_h / ratio > second.bbox[0] and first.bbox[3] + fr_w / ratio > second.bbox[3]:
            return True


def merge_up(first, second):
    bbox = []
    if first.bbox[0] < second.bbox[0]:
        bbox.append(first.bbox[0])
    else:
        bbox.append(second.bbox[0])

    if first.bbox[1] < second.bbox[1]:
        bbox.append(first.bbox[1])
    else:
        bbox.append(second.bbox[1])

    if first.bbox[2] > second.bbox[2]:
        bbox.append(first.bbox[2])
    else:
        bbox.append(second.bbox[2])

    if first.bbox[3] > second.bbox[3]:
        bbox.append(first.bbox[3])
    else:
        bbox.append(second.bbox[3])

    return bbox
