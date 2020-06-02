from collections import defaultdict

class IBM1:
    def __init__(self):
        self.source_vocab = set([])
        self.target_vocab = set([])
        self.trans_prob = defaultdict(lambda: defaultdict(lambda : 0.0))
        self.add_none = False

    def init_prob(self, corpus):
        if self.add_none:
            self.source_vocab.add("<NULL>")
        for x, y in corpus:
            for x_w in x:
                self.source_vocab.add(x_w)
            for y_w in y:
                self.target_vocab.add(y_w)
        prob = 1.0 / len(self.target_vocab)
        for x_w in self.source_vocab:
            for y_w in self.target_vocab:
                self.trans_prob[x_w][y_w] = prob
        # print(prob)
        return

    # def get_obj(self, corpus):
        # P(y|x, len(y)) = \sum_a P(a, y|x, len(y)) = \sum_a p(a|len(x), len(y))p(y|a, x, len(y)) = \prod_j\sum_i trans(y_j,x_i)
        # P(y|x) = P(len(y)|len(x))P(y|x,len(y)) = C(len(x), len(y))P(y|x,len(y))

    def train(self, corpus, iternum):
        self.init_prob(corpus)
        #add noun to source
        #align: align[i][j] P(a_i=j|x,y) target i align to source j

        for iter in range(iternum):
            pesudo_count = defaultdict(lambda: defaultdict(lambda : 0.0))                
            #E-step, Q_i = P(a_i|x,y), update alignment prob
            for x, y in corpus:                
                if self.add_none:
                    x = ["<NULL>"] + x
                align = {}
                for j in range(len(y)):
                    total_norm = 0
                    for i in range(len(x)):
                        align[(j, i)] = self.trans_prob[x[i]][y[j]]
                        total_norm += align[(j, i)]
                    for i in range(len(x)):
                        align[(j, i)] = align[(j, i)] / total_norm
                        pesudo_count[x[i]][y[j]] += align[(j, i)]

            #M-step, E_a P(y,a,x;trans_prob), update translation prob
            for x_w in self.source_vocab:
                total_norm = 0
                for y_w in self.target_vocab:
                    total_norm += pesudo_count[x_w][y_w]
                for y_w in self.target_vocab:
                    self.trans_prob[x_w][y_w] = pesudo_count[x_w][y_w] / total_norm
            print("iter num ", iter + 1, " done.")
        return
    
    def save(self, output_fn):
        lines = []
        lines.append("\t".join([x for x in self.source_vocab]) + "\n")
        lines.append("\t".join([x for x in self.target_vocab]) + "\n")
        lines.append("1\n" if self.add_none else "0\n")
        for x_w, dic in self.trans_prob.items():
            for y_w, prob in dic.items():
                lines.append("{}\t{}\t{}\n".format(x_w, y_w, prob))
        with open(output_fn, mode="w", encoding="utf-8") as fp:
            fp.writelines(lines)
            
        return
    
    def load(self, input_fn):
        ind = 0
        with open(input_fn, mode="r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip("\n")
                if ind == 0:
                    self.source_vocab = set(line.split("\t"))
                elif ind == 1:
                    self.target_vocab = set(line.split("\t"))
                elif ind == 2:
                    self.add_none = True if int(line) == 1 else False
                else:
                    tmp = line.split("\t")
                    self.trans_prob[tmp[0]][tmp[1]] = float(tmp[-1])
                ind += 1
        return

    def align(self, corpus):
        for x, y in corpus:
            print("Source sentence:")
            print(x)
            print("Target sentence:")
            print(y)
            if self.add_none:
                x = ["<NULL>"] + x
            for y_w in y:
                mx_xw = x[0]
                for x_w in x:
                    if self.trans_prob[x_w][y_w] > self.trans_prob[mx_xw][y_w]:
                        mx_xw = x_w
                print("target word: {}\tsource word: {}".format(y_w, mx_xw))
            print("="*30 + "\n")
        return


if __name__ == "__main__":
    zh = []
    with open("./fbis.zh.10k", mode="r", encoding="utf-8") as fp:
        for line in fp:    
            zh.append(line.strip("\n").split())
    eng = []
    with open("./fbis.en.10k", mode="r", encoding="utf-8") as fp:
        for line in fp:
            tmp = line.strip("\n").split()
            eng.append([x.lower() for x in tmp])
    model = IBM1()
    model.train([(eng[i], zh[i]) for i in range(1000)], 10)
    model.save("./model.txt")
#     model.load("./model.txt")
#     model.align([(eng[i], zh[i]) for i in range(10, 12)])
                        
        
    