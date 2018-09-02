def cutoff_lower(image, percent):
    y = image.shape[0]
    cutoff = int(y - (percent * y))
    return image[:cutoff, :]