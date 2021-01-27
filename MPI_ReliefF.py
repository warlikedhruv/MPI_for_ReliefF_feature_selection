from mpi4py import MPI
from mpi_master_slave import Master, Slave
from mpi_master_slave import WorkQueue
import time
import numpy as np
import random
import pandas as pd

class MyApp(object):
    """
    This is my application that has a lot of work to do so it gives work to do
    to its slaves until all the work is done
    """

    def __init__(self, slaves):
        # when creating the Master we tell it what slav[es it can handle
        self.master = Master(slaves)
        # WorkQueue is a convenient class that run slaves on a tasks queue
        self.work_queue = WorkQueue(self.master)

    def terminate_slaves(self):
        """
        Call this to make all slaves exit their run loop
        """
        self.master.terminate_slaves()

    def run(self, tasks=10):
        """
        This is the core of my application, keep starting slaves
        as long as there is work to do
        """
        #
        # let's prepare our work queue. This can be built at initialization time
        # but it can also be added later as more work become available
        #
        df = pd.DataFrame()
        with open('case1.txt') as f:
            p = [int(x) for x in next(f).split()][0]
            n, a, m, t = [int(x) for x in next(f).split()]  # read first line
            array = []
            for line in f:  # read rest of lines
                array.append([float(x) for x in line.split()])
        data = df.append(pd.DataFrame(array, columns=[i for i in range(0, len(array[0]))]), ignore_index=False)
        #print(data)
        df_split = np.array_split(data, p-1)

        for j in range(0,p-1):
            # 'data' will be passed to the slave and can be anything
            list1 = df_split[j].values.tolist()
            self.work_queue.add_work(data=(a, m, t, df.append(pd.DataFrame(list1, columns=[i for i in range(0, len(array[0]))]), ignore_index=False),j))


        #
        # Keeep starting slaves as long as there is work to do
        #
        while not self.work_queue.done():

            #
            # give more work to do to each idle slave (if any)
            #
            self.work_queue.do_work()

            #
            # reclaim returned data from completed slaves
            #
            for slave_return_data in self.work_queue.get_completed_work():
                done, message, i = slave_return_data
                if done:
                    print('NO:',i,'Master: slave finished is task and says "%s"' % message)

            # sleep some time: this is a crucial detail discussed below!
            time.sleep(0.3)


class MySlave(Slave):
    """
    A slave process extends Slave class, overrides the 'do_work' method
    and calls 'Slave.run'. The Master will do the rest
    """

    def __init__(self):
        super(MySlave, self).__init__()

    def do_work(self, data):

        a, m ,t, part_data, index11 = data
        #print(a, m ,t, index11)
        #print(part_data)
        w= []
        hit_index, miss_index = 0, 0

        task_arg = 1
        rows = part_data.shape[0]
        #print(rows)
        for i in range(0,a):
            w.append(0.0)
        for i in range(1,m):


            data_R = random.randrange(0,rows)
            x = part_data.iloc[data_R].tolist()
            class_R = x.pop()
            #print(x)
            miss = 999999
            hit = 999999
            for index, row in part_data.iterrows():  # find nearest hit and miss


                if index == data_R:
                    continue
                else:

                    y = part_data.iloc[index].tolist()
                    class_y = y.pop()
                    #print(y)
                    distance = self.manhattendist(x,y,a)
                    #print(distance)
                    if class_R == class_y:
                        if distance < hit:
                            hit = distance
                            hit_index = index
                    else:
                        if distance < miss:
                            miss = distance
                            miss_index = index

            H = part_data.iloc[hit_index].tolist()
            H.pop()
            M = part_data.iloc[miss_index].tolist()
            M.pop()
            for A in range(0,a):
                w[A] = w[A] - ((self.diff(A,x,H,part_data)) / m) + ((self.diff(A,x,M,part_data))/ m)

        #rank = MPI.COMM_WORLD.Get_rank()
        #name = MPI.Get_processor_name()
        #task, task_arg = data
        #print('  Slave %s rank %d executing "%s" task_id "%d"' % (name, rank, task, task_arg) )

        top={}
        for i in range(0,len(w)):
            top[i]= w[i]

        sorted_dict = {k: v for k, v in sorted(top.items(), key=lambda item: item[1])}


        #print(top)
        w_list = []
        for key in sorted_dict.keys():
            w_list.append(key)

        return (True, w_list[-2:], index11)

    def diff(self,A,I1,I2,data):
        column_data = data[A].tolist()
        maximum = max(column_data)
        minimum = min(column_data)
        value =  (abs(I1[A] - I2[A])) / (maximum - minimum)
        if value > 1:
            print(value)
        return value


    def manhattendist(self,x, y, n):
        sum = 0

        # for each point, finding distance
        # to rest of the point
        for i in range(0,n):

            sum += abs(x[i] - y[i])

        return sum


def main():


    name = MPI.Get_processor_name()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    print('I am  %s rank %d (total %d)' % (name, rank, size) )

    if rank == 0: # Master

        app = MyApp(slaves=range(1, size))
        app.run()
        app.terminate_slaves()

    else: # Any slave

        MySlave().run()

    print('Task completed (rank %d)' % (rank) )

if __name__ == "__main__":
    main()