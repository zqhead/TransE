import codecs
import numpy as np
import copy
import time
import random

entities2vec = {}
relations2vec = {}


def dataloader(file):
    file1 = file + "train.txt"
    file2 = file + "entity2id.txt"
    file3 = file + "relation2id.txt"

    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entities2vec[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relations2vec[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []

    with codecs.open(file1, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue

            h_ = entities2vec[triple[0]]
            t_ = entities2vec[triple[1]]
            r_ = relations2vec[triple[2]]

            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))


class TransE:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0

    def data_initialise(self):
        entityVectorList = {}
        relationVectorList = {}
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            entityVectorList[entity] = entity_vector

        for relation in self.relations:
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                                self.dimension)
            relation_vector = self.normalization(relation_vector)
            relationVectorList[relation] = relation_vector

        self.entities = entityVectorList
        self.relations = relationVectorList

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    def training_run(self, epochs=1000, nbatches=400):

        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for entity in self.entities.keys():
                self.entities[entity] = self.normalization(self.entities[entity]);

            for batch in range(nbatches):
                #
                # if batch == nbatches - 1:
                #     batch_samples = self.triples[batch * batch_size:]
                # else:
                #     batch_samples = self.triples[batch * batch_size: batch * batch_size + batch_size]

                batch_samples = random.sample(self.triples, batch_size)

                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # change the head entity
                        # headindex = np.random.choice(len(self.entities.keys()), size=1, replace=False)[0]
                        # corrupted_sample[0] = self.entities.keys()[headindex]
                        # while corrupted_sample in self.triples: # 防止生成的反例为triples中的其他三元组
                        #     headindex = np.random.choice(len(self.entities.keys()), size=1, replace=False)[0]
                        #     corrupted_sample[0] = self.entities.keys()[headindex]

                        corrupted_sample[0] = random.choice(list(self.entities.keys()))
                        while corrupted_sample in self.triples:  # 防止生成的反例为triples中的其他三元组
                            corrupted_sample[0] = random.choice(list(self.entities.keys()))
                    else:
                        # change the tail entity
                        # tailindex = np.random.choice(len(self.entities.keys()), size=1, replace=False)[0]
                        # corrupted_sample[0] = self.entities.keys()[tailindex]
                        # while corrupted_sample in self.triples:  # 防止生成的反例为triples中的其他三元组
                        #     tailindex = np.random.choice(len(self.entities.keys()), size=1, replace=False)[0]
                        #     corrupted_sample[0] = self.entities.keys()[tailindex]
                        corrupted_sample[1] = random.choice(list(self.entities.keys()))
                        while corrupted_sample in self.triples:  # 防止生成的反例为triples中的其他三元组
                            corrupted_sample[1] = random.choice(list(self.entities.keys()))
                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        with codecs.open("entity_"+self.dimension+"dim_batch"+batch_size, "w") as f1:

            for e in self.entities.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entities[e])))
                f1.write("\n")

        with codecs.open("relation"+self.dimension+"dim_batch"+batch_size, "w", "w") as f2:
            for r in self.relations.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relations[r])))
                f2.write("\n")

    def update_triple_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的entities都不同
        copy_entity = copy.deepcopy(self.entities)
        copy_relation = copy.deepcopy(self.relations)

        for correct_sample, corrupted_sample in Tbatch:

            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[1]]
            relation_copy = copy_relation[correct_sample[2]]

            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[1]]

            correct_head = self.entities[correct_sample[0]]
            correct_tail = self.entities[correct_sample[1]]
            relation = self.relations[correct_sample[2]]

            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[1]]

            # calculate the distance of the triples
            if self.norm == 1:
                correct_distance = norm_l1(correct_head, relation, correct_tail)
                corrupted_distance = norm_l1(corrupted_head, relation, corrupted_tail)

            else:
                correct_distance = norm_l2(correct_head, relation, correct_tail)
                corrupted_distance = norm_l2(corrupted_head, relation, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            if loss > 0:
                self.loss += loss

                correct_gradient = 2 * (correct_head + relation - correct_tail)
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)

                if self.norm == 1:
                    for i in range(len(correct_gradient)):
                        if correct_gradient[i] > 0:
                            correct_gradient[i] = 1
                        else:
                            correct_gradient[i] = -1

                        if corrupted_gradient[i] > 0:
                            corrupted_gradient[i] = 1
                        else:
                            corrupted_gradient[i] = -1

                correct_copy_head -= self.learning_rate*correct_gradient
                relation_copy -= self.learning_rate*correct_gradient
                correct_copy_tail -= -1 * self.learning_rate * correct_gradient

                relation_copy -= -1 * self.learning_rate * corrupted_gradient
                if correct_sample[0] == corrupted_sample[0]:
                    correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                elif correct_sample[1] == corrupted_sample[1]:
                    corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    correct_copy_tail -= self.learning_rate * corrupted_gradient

                # normalise these new embedding vector, instead of all the embedding together
                copy_entity[correct_sample[0]] = self.normalization(correct_copy_head)
                copy_entity[correct_sample[1]] = self.normalization(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity's embedding
                    copy_entity[corrupted_sample[1]] = self.normalization(corrupted_copy_tail)
                elif correct_sample[1] == corrupted_sample[1]:
                    # if corrupted_triples replace the head entity, update the head entity's embedding
                    copy_entity[corrupted_sample[0]] = self.normalization(corrupted_copy_head)
                # the paper mention that the relation's embedding don't need to be normalised
                copy_relation[correct_sample[2]] = relation_copy
                # copy_relation[correct_sample[2]] = self.normalization(relation_copy)

        self.entities = copy_entity
        self.relations = copy_relation


if __name__=='__main__':
    file1 = "FB15k\\"
    entity_set, relation_set, triple_list = dataloader(file1)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (len(entity_set),len(relation_set),len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list)
    transE.data_initialise()
    transE.training_run()








