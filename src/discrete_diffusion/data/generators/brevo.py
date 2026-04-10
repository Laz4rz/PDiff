# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
# This gives the necessary code to generate our [Mano] synthetic arithmetics datasets used in [Physics of language models: Part 4.1]
#


from collections import defaultdict, deque
import random


def generate_multi_token_words(rng, n: int,
    mini_vocab: int = 3,
    min_tlen: int = 5,
    max_tlen: int = 7
):
    def my_sample(length):
        toks = [rng.randint(1, mini_vocab) for _ in range(length)]
        toks[-1] += mini_vocab  # end of word
        return tuple(toks)      # tuples are hashable 

    words = set()
    while len(words) < n:
        length = rng.randint(min_tlen, max_tlen)
        word = my_sample(length)
        words.add(word)

    return [list(word) for word in words]    


class TopoSortDepthStats:
    def __init__(self, n, vocab_size=125, max_in=4):
        self.n = n
        self.vocab_size = vocab_size
        self.max_in = max_in
        self.degree_constraint = 'A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left'
        

    def generate_dag(self, rng):
        nodes = rng.sample(range(1, self.vocab_size + 1), self.n)
        dag = defaultdict(list)
        if self.degree_constraint == 'A_dep_B_for_at_most_4_Bs_and_4_As_with_leaves_on_left':
            out_degree = defaultdict(int)
            leaves = rng.randint(1, (len(nodes)-1)//4+1)
            for i in range(leaves, len(nodes)):
                tgt = nodes[i]
                possible_parents = [src for src in nodes[:i] if out_degree[src] < 4]
                if not possible_parents:
                    continue  # no available parents with capacity
                num_parents = rng.randint(1, min(len(possible_parents), 4))
                #if i!=len(nodes)-1 and rng.randint(0,7)==0:
                #    num_parents = 0
                parents = rng.sample(possible_parents, num_parents)
                for parent in parents:
                    dag[tgt].append(parent)
                    out_degree[parent] += 1
        else:
            assert False, f"I removed other versions; different degree distributions may make the task too easy or degenerate"

        return nodes, dag

    def subtree_from_query(self, dag, query):
        visited = set()
        stack = [query]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for parent in dag.get(node, []):
                    if parent not in visited:
                        stack.append(parent)
        filtered = defaultdict(list)
        for node in visited:
            for parent in dag.get(node, []):
                if parent in visited:
                    filtered[node].append(parent)
        for node in visited:
            _ = filtered[node]
        return filtered

    def topological_sort(self, dag, rng):
        indegree = {node: 0 for node in dag}
        for node in dag:
            for parent in dag[node]:
                indegree[parent] += 1
        queue = [node for node in dag if indegree[node] == 0]
        order = []
        while queue:
            node = queue.pop(rng.randint(0, len(queue) - 1))
            order.append(node)
            for parent in dag[node]:
                indegree[parent] -= 1
                if indegree[parent] == 0:
                    queue.append(parent)
        order.reverse()
        return order

    def compute_graph_depth(self, dag, query):
        distance = {query: 0}
        queue = deque([query])
        while queue:
            node = queue.popleft()
            for parent in dag[node]:
                if parent not in distance:
                    distance[parent] = distance[node] + 1
                    queue.append(parent)
        leaves = [node for node in dag if len(dag[node]) == 0]
        if not leaves:
            return 0
        return min(distance.get(leaf, float('inf')) for leaf in leaves if leaf in distance)

    def generate_sample(self, rng):
        nodes, dag = self.generate_dag(rng)

        start_index = max(len(nodes) * 3 // 4, len(nodes) - 1)  # safe even if len(nodes) == 2
        candidate_nodes = nodes[start_index:]
        nonzero_degree_nodes = [node for node in candidate_nodes if len(dag[node]) > 0]
        query = rng.choice(nonzero_degree_nodes)

        subdag = self.subtree_from_query(dag, query)
        topo = self.topological_sort(subdag, rng)
        depth = self.compute_graph_depth(subdag, query)
        return dag, topo, depth


    def generate_tokens(self, rng, multi=False):
        dag, topo, depth = self.generate_sample(rng)
        query = topo[-1]

        if multi:
            all_node_ids = sorted(set(dag.keys()) | {p for ps in dag.values() for p in ps})
            id_to_index = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
            index_to_id = {idx: node_id for node_id, idx in id_to_index.items()}

            word_list = generate_multi_token_words(rng,
                n=len(all_node_ids),
                mini_vocab=4,
                min_tlen=2,
                max_tlen=4,
            )

            word_map = {node_id: word_list[id_to_index[node_id]] for node_id in all_node_ids}
    
            
        edges = [(p, c) for c, ps in dag.items() for p in ps]
        rng.shuffle(edges)


        tokens = [bos_token_id]
        for p, c in edges:
            if multi:
                tokens += word_map[p] + word_map[c]
            else:
                tokens += [p, c]
        if multi:
            tokens += [bos_token_id-2] + word_map[query] + [bos_token_id-1]
        else:
            tokens += [bos_token_id-2, query, bos_token_id-1]
        list_label = [0] * (len(tokens)-1)
        token_type = [-1, -2-depth] + [0] * (len(tokens)-2)
        fake_depth = depth if depth<=9 else 9
        if multi:
            first = True
            for node in topo:
                tokens += word_map[node]
                if first: 
                    token_type += [fake_depth+1] * len(word_map[node])
                    first = False
                else:
                    token_type += [0] * len(word_map[node])
        else:
            tokens += topo
            token_type += [fake_depth+1] + [0] * (len(topo)-1)  # only check if first token is correct, for training illustration only
        tokens += [eos_token_id]
        token_type += [0]
        list_label += [1] * (len(tokens) - len(list_label))
        assert len(tokens) == len(list_label) and len(list_label) == len(token_type), f"{len(tokens)} {len(list_label)} {len(token_type)}"
        return tokens, token_type, list_label, depth
       
    @staticmethod
    def parse_tokens(tokens):
        tokens = [a for a in tokens if a != pad_token_id]
        if not tokens:
            return False, None, None
        if tokens[0] != bos_token_id or tokens[-1] != eos_token_id:
            return False, None, None
        try:
            idx_query = tokens.index(bos_token_id-2)
            idx_answer = tokens.index(bos_token_id-1)
        except ValueError:
            return False, None, None

        edge_tokens = tokens[1:idx_query]
        if len(edge_tokens) % 2 != 0:
            return False, None, None

        edges = [(edge_tokens[i], edge_tokens[i + 1]) for i in range(0, len(edge_tokens), 2)]
        query = tokens[idx_query + 1]
        topo = tokens[idx_answer + 1 : -1]

        # Build DAG: child ← [parents]
        dag = defaultdict(list)
        for parent, child in edges:
            dag[child].append(parent)

        # Step 1: compute reachable nodes (reverse DFS from query)
        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        # Step 2: validate all nodes in topo are reachable from query
        if set(topo) != reachable:
            return False, query, topo

        # Step 3: validate topological order
        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, topo
            seen.add(node)

        # print("Correctness explanation:")
        # print(dag)
        # print("query=",query)
        # print(topo)
        # print("End of explanation")


        return True, query, topo


    # Can use the following trivial code to evaluate a model's output --- topsort answer is not unique
    @staticmethod
    def parse_tokens_multi(tokens, mini_vocab=4):
        tokens = [a for a in tokens if a != pad_token_id]
        if not tokens:
            return False, None, None
        if tokens[0] != bos_token_id or tokens[-1] != eos_token_id:
            return False, None, None
        try:
            idx_query = tokens.index(bos_token_id-2)
            idx_answer = tokens.index(bos_token_id-1)
        except ValueError:
            return False, None, None

        def split_words(token_seq):
            words = []
            word = []
            for tok in token_seq:
                word.append(tok)
                if mini_vocab < tok <= 2 * mini_vocab:
                    words.append(tuple(word))
                    word = []
            return words

        # 1. Parse all sections
        edge_tokens = tokens[1:idx_query]
        query_tokens = tokens[idx_query + 1 : idx_answer]
        answer_tokens = tokens[idx_answer + 1 : -1]

        edge_words = split_words(edge_tokens)
        query_words = split_words(query_tokens)
        if len(query_words) != 1:
            return False, None, None
        query_word = tuple(query_words[0])
        answer_words = split_words(answer_tokens)

        if len(edge_words) % 2 != 0:
            return False, None, None

        # 2. Assign unique IDs to each word
        all_words = set(edge_words + [query_word] + answer_words)
        word_to_id = {w: i for i, w in enumerate(sorted(all_words))}
        id_to_word = {i: w for w, i in word_to_id.items()}

        # 3. Reconstruct DAG: child ← [parents]
        dag = defaultdict(list)
        for i in range(0, len(edge_words), 2):
            p, c = edge_words[i], edge_words[i + 1]
            dag[word_to_id[c]].append(word_to_id[p])

        query = word_to_id[query_word]
        topo = [word_to_id[w] for w in answer_words]

        # 4. Get reachable nodes from query
        reachable = set()
        stack = [query]
        while stack:
            node = stack.pop()
            reachable.add(node)
            for parent in dag.get(node, []):
                if parent not in reachable:
                    stack.append(parent)

        if set(topo) != reachable:
            return False, query, [id_to_word[i] for i in topo]

        # 5. Validate topological order
        seen = set()
        for node in topo:
            for parent in dag.get(node, []):
                if parent not in seen:
                    return False, query, [id_to_word[i] for i in topo]
            seen.add(node)

        return True, query, [id_to_word[i] for i in topo]

    
rng = random.Random(42)

# NOTE: during testing, we enforce n = N and the code is NOT provided here

def _sample_task_size(N, rng_obj=None):
    if rng_obj is None:
        rng_obj = rng
    _distribution_list_to_choose = list(range(3, N+1))
    power, bias = 1, pow(N, 0.5)
    p = [1.0 / (pow(i, power) + bias + 1e-12) for i in _distribution_list_to_choose]
    _distribution_p = [1.0 * x / sum(p) for x in p]
    return rng_obj.choices(_distribution_list_to_choose, weights=_distribution_p)[0]


def topsort_data(N, multi=False, enforce_n=False, rng_obj=None):
    if rng_obj is None:
        rng_obj = rng
    if not enforce_n:
        n = _sample_task_size(N, rng_obj=rng_obj)
    else:
        n = N

    topo = TopoSortDepthStats(n, vocab_size=N)
    text, token_type, list_labels, depth = topo.generate_tokens(rng_obj, multi=multi)
    
    ## token_type is for my own reference, not used for training
    ## list_labels = 0 or 1, where 1's are for answer tokens. We pretrain only on those answer tokens --- pretraining with all the tokens yield similar results, but is a factor slower.
    return {0:text, 1:token_type, 'label':list_labels}

def topsort_data_raw(N, rng_obj=None):
    if rng_obj is None:
        rng_obj = rng
    n = _sample_task_size(N, rng_obj=rng_obj)
    topo = TopoSortDepthStats(n, vocab_size=N)
    dag, topo_order, depth = topo.generate_sample(rng_obj)
    query = topo_order[-1]
    subdag = topo.subtree_from_query(dag, query)
    edges = [(parent, child) for child, parents in dag.items() for parent in parents]

    # Non-tokenized structure: graph + query + valid topological answer for query's ancestor subgraph.
    return {
        "n": n,
        "dag": {node: list(parents) for node, parents in dag.items()},
        "subdag": {node: list(parents) for node, parents in subdag.items()},
        "edges": edges,
        "query": query,
        "topo": topo_order,
        "depth": depth,
    }


# Keep special token ids compact and just above the largest N used in examples.
max_example_node_id = 110
bos_token_id = max_example_node_id + 3
eos_token_id = bos_token_id - 1  # keeps `bos-1` delimiter compatible
pad_token_id = bos_token_id + 1
mask_token_id = bos_token_id + 2

if __name__ == "__main__":
    print(topsort_data(N=110))  # used as our Brevo1 tasks
    print(topsort_data(N=90))
    print(topsort_data(N=70))

    print(topsort_data(N=50, multi=True))  # used as our Brevo2 tasks
    print(topsort_data(N=40, multi=True))
    print(topsort_data(N=30, multi=True))
