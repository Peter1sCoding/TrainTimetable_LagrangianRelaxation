# 节点类，储存流入与流出该节点的弧集

class Node():
    def __init__(self, sta, t):
        self.sta_located = sta
        self.t_located = t
        self.in_arcs = {}  # 流入该节点的弧集，以trainNo为key，弧为value
        self.out_arcs = {}  # 流出该节点的弧集，以trainNo为key, 弧为value
        self.incompatible_arcs = []  # 该节点对应资源占用<=1的约束中，不相容弧的集合，以trainNo为索引，子字典以arc_length为key
        self.multiplier = 0  # 该节点对应约束的拉格朗日乘子
        self.name = [self.sta_located, self.t_located]
        self.isOccupied = False # 可行解中，该节点是否已经被占据

    def __repr__(self):
        return "Node: " + str(self.sta_located) + " at time " + str(self.t_located)

    def __str__(self):
        return "Node: sta_" + self.sta_located + ";" + "t_" + self.t_located

    def associate_with_incoming_arcs(self, train):
        '''
        associate node with train arcs, add incoming arcs to nodes
        :param train:
        :return:
        '''
        sta_node = self.sta_located
        t_node = self.t_located

        if sta_node not in train.v_staList:  # 若该车不经过该站，直接退出
            return -1

        # associate incoming arcs
        # train arc structure: key：[dep, arr], value为弧集字典(key: [t], value: arc字典, key为arc_length)
        if sta_node != train.v_staList[0]:  # 不为第一站，则拥有上一站
            preSta = train.v_staList[train.v_staList.index(sta_node) - 1]  # 已经考虑列车停站情况的车站集
            curSta = sta_node
            cur_arcs = train.arcs[preSta, curSta]  # 这个区间/车站的所有弧
            for start_t in cur_arcs.keys():
                arcs_from_start_t = cur_arcs[start_t]  # 从上一节点在start_t伸出来的arc，包括区间的一个arc和停站弧的若干个arc
                for arc_length, arc_var in arcs_from_start_t.items():
                    if arc_length + start_t == t_node:  # 若该弧流入该节点
                        # 若不包含该车的弧列表，则先生成弧列表
                        if train.traNo not in self.in_arcs.keys():
                            self.in_arcs[train.traNo] = {}
                        self.in_arcs[train.traNo][arc_length] = arc_var

    def associate_with_outgoing_arcs(self, train):
        '''
        associate node with train arcs, add outgoing arcs to nodes
        :param train:
        :return:
        '''
        sta_node = self.sta_located
        t_node = self.t_located # 所在点

        if sta_node not in train.v_staList or sta_node == train.v_staList[-1]:  # 列车径路不包含该站 或 该站为最后一站，则都不会有流出弧
            return -1

        curSta = sta_node
        nextSta = train.v_staList[train.v_staList.index(sta_node) + 1]
        cur_arcs = train.arcs[curSta, nextSta]
        if sta_node == train.v_staList[-2]:
            b = 0
        if t_node in cur_arcs.keys():  # 如果点t在列车区间/车站弧集当中，source node就是-1
            self.out_arcs[train.traNo] = {}
            for arc_length, arc_var in cur_arcs[t_node].items():
                self.out_arcs[train.traNo][arc_length] = arc_var

