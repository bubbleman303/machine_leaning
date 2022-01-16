def get_value(obj, x, y, x1=None, y1=None):
    if x1 is None:
        return obj.Range((x, y)).value
    return obj.Range((x, y), (x1, y1)).value


def set_value(obj, x, y, val):
    obj.Range((x, y)).value = val
