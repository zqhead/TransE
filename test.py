import json
import operator # operator模块输出一系列对应Python内部操作符的函数
import numpy as np
import codecs


from transE import dataloader,entities2id,relations2id


def test_data_loader(entity_embedding_file, relation_embedding_file, test_data_file):
    print("load data...")
    file1 = entity_embedding_file
    file2 = relation_embedding_file
    file3 = test_data_file

    entity_dic = {}
    relation_dic = {}
    triple_list = []

    with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity_dic[line[0]] = json.loads(line[1])

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation_dic[line[0]] = json.loads(line[1])

    with codecs.open(file3, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            head = entities2id[triple[0]]
            tail = entities2id[triple[1]]
            relation = relations2id[triple[2]]

            triple_list.append([head, tail, relation])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_dic.keys()), len(relation_dic.keys()), len(triple_list)))

    return entity_dic, relation_dic, triple_list

class testTransE:
    def __init__(self, entities_dict, relations_dict, test_triple_list, train_triple_list, filter_triple=False, norm=1):
        self.entities = entities_dict
        self.relations = relations_dict
        self.test_triples = test_triple_list
        self.train_triples = train_triple_list
        self.filter = filter_triple
        self.norm = norm

        self.mean_rank = 0
        self.hit_10 = 0

    def test_run(self):
        hits = 0
        rank_sum = 0
        num = 0
        for triple in self.test_triples:
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            for entity in self.entities.keys():
                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in self.train_triples:
                        continue
                head_embedding = self.entities[entity]
                tail_embedding = self.entities[triple[1]]
                relation_embedding = self.relations[triple[2]]
                distance = self.distance(head_embedding, relation_embedding, tail_embedding)
                rank_head_dict[tuple(head_triple)] = distance

            for tail in self.entities.keys():
                tail_triple = [triple[0], tail, triple[2]]
                if self.filter:
                    if tail_triple in self.train_triples:
                        continue
                head_embedding = self.entities[tail_triple[0]]
                tail_embedding = self.entities[tail_triple[1]]
                relation_embedding = self.relations[tail_triple[2]]
                distance = self.distance(head_embedding, relation_embedding, tail_embedding)
                rank_tail_dict[tuple(head_triple)] = distance

            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            # calculate the mean_rank and hit_10
            # head data
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            # tail rank
            for i in range(len(rank_tail_sorted)):
                if triple[1] == rank_tail_sorted[i][0][1]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))

        return self.hit_10,  self.mean_rank


    def distance(self, h, r, t):
        head = np.array(h)
        relation = np.array(r)
        tail = np.array(t)
        d = head + relation - tail
        if self.norm == 1:
            return np.sum(np.fabs(d))
            # return np.linalg.norm(d, ord=1)
        else:
            return np.sum(np.square(d))
            # return np.linalg.norm(d, ord=2)


if __name__ == "__main__":
    _, _, train_triple = dataloader("FB15k\\")

    entity_dict, relation_dict, test_triple = test_data_loader("entity_vector_50dim",
                                                               "relation_vector_50dim", "FB15k\\test.txt")

    test = testTransE(entity_dict, relation_dict, test_triple, train_triple, filter_triple=False, norm=2)
    hit10, mean_rank = test.test_run()
    print("raw entity hits@10: ", hit10)
    print("raw entity meanrank: ",mean_rank)

    test2 = testTransE(entity_dict, relation_dict, test_triple, train_triple, filter_triple=True, norm=2)
    filter_hit10, filter_mean_rank = test2.test_run()
    print("filter entity hits@10: ", filter_hit10)
    print("filter entity meanrank: ", filter_mean_rank)