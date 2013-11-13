

class BTree:
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

    def insert(self, d):
        if self.data == None:
            self.data = d
            self.left = BTree()
            self.right = BTree()
            return
        else:
            if d <= self.data:
                self.left.insert(d)
            else:
                self.right.insert(d)

    def show(self):
        if self.data == None:
            pass
        else:
            self.left.show()
            print self.data
            self.right.show()



if __name__ == "__main__":

    xs = [3, 1, 4, 7, 2]

    B = BTree()
    for x in xs:
        B.insert(x)

    B.show()

    
