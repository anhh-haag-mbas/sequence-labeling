class Mapper:
    """
    Takes in a list of elements and creates creates a bidirectional mapping between each element and an index.
    """
    def __init__(self, elements, default = None):
        self.default = default
        self.i2x = list(set(elements))
        self.i2x.sort()
        if default: self.i2x.insert(0, default)
        self.x2i = {x:i for i,x in enumerate(self.i2x)}

    def element(self, index):
        if index >= len(self) and self.default:
            return self.i2x[0]
        return self.i2x[index]

    def element_list(self, indexes):
        return [self.element(i) for i in indexes]

    def index(self, element):
        if self.default:
            return self.x2i.get(element, 0)
        return self.x2i[element]

    def index_list(self, elements):
        return [self.index(e) for e in elements]

    def __len__(self):
        return len(self.i2x)
