# Written by: https://github.com/pum1k
# and: https://github.com/Saniel0

import queue

def sort_blobs(data):

	# data = [(i, j) for i in range(7) for j in range(5)]
	# data = [(i, j) for i in range(5) for j in range(7)]

	edges = {frozenset([a, b]) for a in data for b in data if a != b}

	def cost(edge):
		(x1, y1), (x2, y2) = edge
		return (x1 - x2) ** 2 + (y1 - y2) ** 2

	edges = sorted(edges, key=cost, reverse=True)

	print(data)
	print(f"{len(data)=}")
	print(f"{len(edges)=}")

	degrees = {a: len(data) - 1 for a in data}
	print(degrees)

	allowed_degrees = {
		2: 4,
		3: 16,
		4: 15,
	}

	def _calc_updates_impl(updates, vertex):
		deg = degrees[vertex]
		if deg <= 5:
			updates[deg - 1] -= 1
		if deg <= 4:
			updates[deg] += 1

	def get_updates(vertices: list):
		updates = {2: 0, 3: 0, 4: 0}
		for vertex in vertices:
			_calc_updates_impl(updates, vertex)
		return updates

	def apply_updates(updates):
		for k in [2, 3, 4]:
			allowed_degrees[k] += updates[k]

	def can_remove(edge):
		a, b = edge
		if degrees[a] <= 2 or degrees[b] <= 2:
			return False, None
		updates = get_updates(edge)
		for k in [2, 3, 4]:
			s = 0
			for j in range(2, k + 1):
				s += allowed_degrees[j] + updates[j]
			if s < 0:
				return False, None
		return True, updates

	def bfs():
		start = list(degrees.keys())[list(degrees.values()).index(2)]
		parents = {start: None}
		q = queue.Queue()
		q.put(start)
		while not q.empty():
			current = q.get()
			print(current, degrees[current])
			if degrees[current] == 2 and current != start:
				graph_side = [current]
				for _ in range(4):
					current = parents[current]
					graph_side.append(current)
				return graph_side
			else:
				for node in data:
					if frozenset([node, current]) in kept and node not in parents:
						parents[node] = current
						q.put(node)

	def calc_grid(graph_edge):
		visited = set(graph_edge)
		grid = [graph_edge]
		for _ in range(6):
			line = []
			prev = grid[-1]
			for x in prev:
				for node in data:
					if frozenset([x, node]) in kept and node not in visited:
						line.append(node)
						visited.add(node)
						break
			grid.append(line)
		return grid

	def calc_idxs(grid):
		k = len(grid[0])
		out = {}
		for i, row in enumerate(grid):
			for j, val in enumerate(row):
				out[i * k + j] = val
		return out

	kept = []
	for edge in edges:
		a, b = edge
		success, updates = can_remove(edge)
		if not success:
			kept.append(edge)
		else:
			degrees[a] -= 1
			degrees[b] -= 1
			apply_updates(updates)

	print(f"{len(kept)=}")
	print(kept)

	out_map = {x: i for i, x in enumerate(data)}

	# print()
	# print("graph {")
	# for edge in kept:
	#     a, b = edge
	#     print(f"{out_map[a]} -- {out_map[b]}")
	# print("}")

	# print("=" * 80)
	# for key in data:
	#     print(f"{key} ~ {degrees[key]}")
	# print("=" * 80)

	graph_edge = bfs()
	print(graph_edge)
	grid = calc_grid(graph_edge)
	print("=" * 60)
	print(*grid, sep="\n")
	print("=" * 60)
	out = calc_idxs(grid)
	print(out)

	return out