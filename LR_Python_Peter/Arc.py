## 弧集合
class Arc():
    def __init__(self, trainBelong, staBelong_preSta, staBelong_next, timeBelong_pre, timeBelong_next, arc_length):
        self.trainBelong = trainBelong
        self.staBelong_pre = staBelong_preSta
        self.staBelong_next = staBelong_next
        self.timeBelong_pre = timeBelong_pre
        self.timeBelong_next = timeBelong_next
        self.arc_length = arc_length
        self.isChosen_LR = 0 # 0为没选，1为选
        self.before_occupy_dep = 1  # 前1
        self.after_occupy_dep = 2  # 后2
        self.before_occupy_arr = 1  # 前1
        self.after_occupy_arr = 2  # 后2
        self.node_occupied = []  # 该弧参与的约束的集合，约束此时已经转为node-related，所以以node的乘子来表示约束的乘子

    def __repr__(self):
        pre_t = str(self.timeBelong_pre)
        next_t = str(self.timeBelong_next)
        return self.trainBelong + ": " + self.staBelong_pre + "(" + pre_t + ") => " + self.staBelong_next + "(" + next_t \
               + ")"
