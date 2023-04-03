class TreeNode(object):
    def __init__(self, index, state, name):
        self.index = index
        self.state = state
        self.name = name
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def add_child_to(self, index, newNode):
        targetNode = self.find_node(index)
        targetNode.add_child(newNode)

    def find_node(self, index):
        foundNode = None
        # init node
        if self.index == index:
            return self
        # else
        for childNode in self.children:
            if childNode.index == index:
                return childNode
            foundNode = childNode.find_node(index)
            # leaf node
            if foundNode != None:
                return foundNode

    def update_state(self):
        if self.state == '':
            self.state = 'current'
        else:
            self.state = 'none'
            for child in self.children:
                child.update_state()
        return self

    def dict_to_tree(self, childNodeList):
        for childNode in childNodeList:
            newNode = TreeNode(index = childNode['index'], state = childNode['state'], name = childNode['name'])
            newNode = newNode.dict_to_tree(childNode['children'])
            self.add_child(newNode)
        return self

    def tree_to_dict(self):
        treeDict = dict()
        children = []
        for child in self.children:
            childDict = child.tree_to_dict()
            children.append(childDict)
        treeDict.update({'index': self.index, 'state': self.state, 'name': self.name, 'children': children})
        return treeDict