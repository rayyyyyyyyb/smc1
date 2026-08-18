class Object:
    def __init__(self,I):
        self.I=I
        self.Start=[]
        self.End=[]
        self.T=[]
        self.assign_for=[]

    def _add(self,S,E,obs,t):
        self.Start.append(S) #记录新任务开始时间
        self.End.append(E)   #记录新任务结束时间
        self.Start.sort()    #保持开始时间有序
        self.End.sort()      #保持结束时间有序
        self.T.append(t)
        self.assign_for.insert(self.End.index(E),obs) #根据结束时间的顺序插入任务对象

    #local search
    #计算空闲时间
    def idle_time(self):
        Idle=[]
        try:
            if self.Start[0]!=0:
                Idle.append([0,self.Start[0]])
            K=[[self.End[i],self.Start[i+1]] for i in range(len(self.End)) if self.Start[i+1]-self.End[i]>0] #若 Start[i+1] > End[i]，说明任务 i 和 i+1 之间存在空闲时间段
            Idle.extend(K) #将空闲时间段计入列表
        except:
            pass
        return  Idle