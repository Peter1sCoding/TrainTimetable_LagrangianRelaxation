# Here defines the info about the trains
import copy

from Arc import *
## 所有的arc取值就是0或1，即选或不选

class Train():
    def __init__(self, traNo, dep_LB, dep_UB):
        '''
        construct
        :param traNo:
        :param dep_LB:
        :param dep_UB:
        '''
        self.traNo = traNo  # 列车车次
        self.dep_LB = dep_LB  # 始发时间窗下界
        self.dep_UB = dep_UB  # 始发时间窗上界
        self.arcs = {}  # 内含弧两个边界点的key：[dep, arr], value为弧集字典(key: [t], value: arc字典, key为arc_length) 三层字典嵌套: dep-arr => t => span
        self.stop_addTime = 3  # 停车附加时分
        self.start_addTime = 2  # 起车附加时分
        self.min_dwellTime = 2  # 最小停站时分
        self.max_dwellTime = 6  # 最大停站时分
        self.secTimes = {}
        self.right_time_bound = {} # 各站通过线路时间窗和列车始发时间窗综合确定的右侧边界
        self.depSta = None
        self.arrSta = None
        self.v_staList = [] # dual stations
        self.staList = [] # actual stations
        self.linePlan = {}  # 开行方案字典
        self.opt_path_LR = None # LR 中的最短路径
        self.last_opt_path_LR = None
        self.opt_cost_LR = 0
        self.feasible_path = None # 可行解中的最短路径
        self.last_feasible_path = None # 上一个可行解的最短路径，用于置0
        self.feasible_cost = 0
        self.timetable = {} # 以virtual station为key，存int值
        self.speed = None # 列车速度，300,350

    def __repr__(self):
        return self.traNo

    def init_traStaList(self, allStaList):
        '''
        create train staList, include s_, _t， only contains nodes associated with this train
        :param allStaList:
        :return:
        '''
        for sta in allStaList:
            if sta in self.linePlan.keys():
                self.staList.append(sta)
        self.v_staList.append('s_')
        for i in range(len(self.staList)):
            if i != 0:
                self.v_staList.append('_' + self.staList[i])
            if i != len(self.staList) - 1:  # 若不为实际车站的最后一站，则加上sta_
                self.v_staList.append(self.staList[i] + '_')
        self.v_staList.append('_t')

    def create_arcs_LR(self, secTimes, TimeSpan):
        self.depSta = self.staList[0]
        self.arrSta = self.staList[-1]
        self.secTimes = secTimes
        self.truncate_train_time_bound(TimeSpan)
        '''
        create train arcs
        :param v_staList:
        :param secTimes:
        :param model:
        :return:
        '''
        minArr = self.dep_LB  # for curSta(judge by dep)
        '''
        create arcs involving node s
        '''
        self.arcs['s_', self.staList[0] + '_'] = {}
        self.arcs['s_', self.staList[0] + '_'][-1] = {}  # source node流出弧, 只有t=-1，因为source node与时间无关
        for t in range(minArr, self.right_time_bound[self.v_staList[1]]):
            self.arcs['s_', self.staList[0] + '_'][-1][t] = Arc(self.traNo, 's_', self.staList[0] + '_', -1, t, 0)
            # 声明弧长为t，实际length为0
        '''
        create arcs between real stations
        '''
        for i in range(len(self.staList) - 1):
            curSta = self.staList[i]
            nextSta = self.staList[i + 1]
            # virtual dual stations
            curSta_dep = curSta + "_"
            nextSta_arr = "_" + nextSta
            nextSta_dep = nextSta + "_"

            secRunTime = secTimes[curSta, nextSta]  # 区间运行时分

            # 创建两个弧, 一个运行弧，一个停站弧
            '''
            curSta_dep-->nextSta_arr区间运行弧
            '''
            self.arcs[curSta_dep, nextSta_arr] = {}
            secRunTime += self.stop_addTime  # 添加停车附加时分

            if self.linePlan[curSta] == 1:  # 本站停车， 加起停附加时分
                secRunTime += self.start_addTime
            # 设置d-a的区间运行弧
            for t in range(minArr, self.right_time_bound[curSta_dep]):
                if t + secRunTime >= self.right_time_bound[nextSta_arr]:  # 范围为0 => TimeSpan - 1
                    break
                self.arcs[curSta_dep, nextSta_arr][t] = {}  # dep-arr在node t的弧集，固定区间运行时分默认只有一个元素
                self.arcs[curSta_dep, nextSta_arr][t][secRunTime] = Arc(self.traNo, curSta_dep, nextSta_arr, t, t+secRunTime, secRunTime)
            # update cur time window
            minArr += secRunTime

            '''
            nextSta_arr-->nextSta_dep车站停站弧
            '''
            if i + 1 == len(self.staList) - 1:  # 若停站, 但已经是最后一个站了，不需停站弧
                break

            self.arcs[nextSta_arr, nextSta_dep] = {}
            if self.linePlan[nextSta] == 1: # 该站停车，创建多个停站时间长度的停站弧
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    if t + self.min_dwellTime >= self.right_time_bound[nextSta_dep]:  # 当前t加上最短停站时分都超了，break掉
                        break
                    self.arcs[nextSta_arr, nextSta_dep][t] = {}
                    for span in range(self.min_dwellTime, self.max_dwellTime):
                        if t + span >= self.right_time_bound[nextSta_dep]:
                            break
                        self.arcs[nextSta_arr, nextSta_dep][t][span] = Arc(self.traNo, nextSta_arr, nextSta_dep, t, t+span, span)
            else: # 该站不停车，只创建一个竖直弧，长度为0
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    self.arcs[nextSta_arr, nextSta_dep][t] = {}
                    self.arcs[nextSta_arr, nextSta_dep][t][0] = Arc(self.traNo, nextSta_arr, nextSta_dep, t, t, 0)
            # update cur time window
            minArr += self.min_dwellTime

        '''
        create arcs involving node t
        '''
        self.arcs['_' + self.staList[-1], '_t'] = {}
        for t in range(minArr, self.right_time_bound[self.v_staList[-2]]):
            self.arcs['_' + self.staList[-1], '_t'][t] = {}  # dep-arr在node t的弧集，固定区间运行时分默认只有一个元素
            self.arcs['_' + self.staList[-1], '_t'][t][0] = Arc(self.traNo, self.staList[-1], '_t', t, -1, 0)



    def update_arc_chosen(self):
        '''
        通过获取的opt_path，将路径中包含的弧的 isChosen 属性更新
        :return:
        '''
        # 先把上一轮的path清零
        # 内含弧两个边界点的key：[dep, arr], value为弧集字典(key: [t], value: arc字典, key为arc_length) 三层字典嵌套: dep-arr => t => span
        if self.last_opt_path_LR is not None: # 第一轮循环还没有，不用设为0
            for node_id in range(1, len(self.last_opt_path_LR.node_passed) - 2):
                node_name = self.last_opt_path_LR.node_passed[node_id]
                next_node_name = self.last_opt_path_LR.node_passed[node_id + 1]
                self.arcs[node_name[0], next_node_name[0]][node_name[1]][next_node_name[1] - node_name[1]].isChosen_LR = 0

        # 再把这一轮的设为1
        # 内含弧两个边界点的key：[dep, arr], value为弧集字典(key: [t], value: arc字典, key为arc_length) 三层字典嵌套: dep-arr => t => span
        for node_id in range(1, len(self.opt_path_LR.node_passed) - 2):
            node_name = self.opt_path_LR.node_passed[node_id]
            next_node_name = self.opt_path_LR.node_passed[node_id + 1]
            self.arcs[node_name[0], next_node_name[0]][node_name[1]][next_node_name[1] - node_name[1]].isChosen_LR = 1
        self.last_opt_path_LR = copy.deepcopy(self.opt_path_LR)  #将上一个最优记录下来，下一次先把上一次路径的chosen清零

    def truncate_train_time_bound(self, TimeSpan):
        right_bound_by_sink = [] # 从总天窗时间右端反推至该站的右侧边界，按运行最快了算
        accum_time = 0
        right_bound_by_sink.append(TimeSpan - accum_time)  # 最后一站的到达
        for sta_id in range(len(self.staList) - 1, 0, -1):
            accum_time += self.secTimes[self.staList[sta_id - 1], self.staList[sta_id]]
            right_bound_by_sink.append(TimeSpan - accum_time)
            if sta_id != 1: # 最后一站不用加上停站时分了
                if self.linePlan[self.staList[sta_id - 1]] == 1: #若停站了则加一个2
                    accum_time += self.min_dwellTime
                else:
                    accum_time += 0
                right_bound_by_sink.append(TimeSpan - accum_time)
        right_bound_by_sink = list(reversed(right_bound_by_sink))


        right_bound_by_dep = []
        right_bound_by_dep.append(self.dep_UB) # 第一个站
        accum_time = self.dep_UB
        for sta_id in range(0, len(self.staList) - 1): # 最后一个站不考虑
            accum_time += self.secTimes[self.staList[sta_id], self.staList[sta_id + 1]]
            right_bound_by_dep.append(accum_time)
            if sta_id != len(self.staList) - 2: # 最后一个区间，不加停站时分
                accum_time += self.min_dwellTime
                right_bound_by_dep.append(accum_time)

        for sta in self.v_staList:
            if sta == self.v_staList[-1] or sta == self.v_staList[0]:
                continue
            right_bound_dep = right_bound_by_dep[self.v_staList.index(sta) - 1]
            right_bound_sink = right_bound_by_sink[self.v_staList.index(sta) - 1]
            self.right_time_bound[sta] = min(right_bound_dep, right_bound_sink)























        
