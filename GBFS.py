import math
import time
from dataclasses import dataclass, field
from functools import lru_cache
from queue import PriorityQueue
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

Number = Union[int, float]


@dataclass(eq=False)
class Node:
    x: int
    y: int
    cost: Number = 0
    parent: "Node" = None

    def __sub__(self, other) -> int:
        if isinstance(other, Node):
            return abs(self.x - other.x) + abs(self.y - other.y)
        elif isinstance(other, (tuple, list)):
            return abs(self.x - other[0]) + abs(self.y - other[1])
        raise ValueError("other must be coordinates or Node")

    def __add__(self, other: Union[tuple, list]) -> "Node":
        x = self.x + other[0]
        y = self.y + other[1]
        cost = self.cost + math.sqrt(other[0] ** 2 + other[1] ** 2)
        return Node(x, y, cost, self)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, (tuple, list)):
            return self.x == other[0] and self.y == other[1]
        return False

    def __le__(self, other: "Node"):
        return self.cost <= other.cost

    def __lt__(self, other: "Node"):
        return self.cost < other.cost

    def __hash__(self) -> int:
        return hash((self.x, self.y))


@dataclass
class SetQueue:
    queue: set[Node] = field(default_factory=set)

    def __bool__(self):
        return bool(self.queue)

    def __contains__(self, item):
        return item in self.queue

    def __len__(self):
        return len(self.queue)

    # PriorityQueue
    def get(self):
        node = min(self.queue)  # O(n)?
        self.queue.remove(node)  # O(1)
        return node

    def put(self, node: Node):
        if node in self.queue:  # O(1)
            qlist = list(self.queue)
            idx = qlist.index(node)  # O(n)
            if node.cost < qlist[idx].cost:
                self.queue.remove(node)
                self.queue.add(node)
        else:
            self.queue.add(node)  # O(1)

    def empty(self):
        return len(self.queue) == 0


@dataclass
class ListQueue:
    queue: list[Node] = field(default_factory=list)

    def __bool__(self):
        return bool(self.queue)

    def __contains__(self, item):
        return item in self.queue

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, idx):
        return self.queue[idx]

    def append(self, node: Node):
        self.queue.append(node)  # O(1)

    def pop(self, idx=-1):
        return self.queue.pop(idx)  # O(1) ~ O(n)

    # PriorityQueue
    def get(self):
        idx = self.queue.index(min(self.queue))  # O(n) + O(n)
        return self.queue.pop(idx)  # O(1) ~ O(n)

    def put(self, node: Node):
        if node in self.queue:  # O(n)
            idx = self.queue.index(node)  # O(n)
            if node.cost < self.queue[idx].cost:
                self.queue[idx].cost = node.cost  # O(1)
                self.queue[idx].parent = node.parent  # O(1)
        else:
            self.queue.append(node)  # O(1)

    def empty(self):
        return len(self.queue) == 0


class PriorityQueuePro(PriorityQueue):

    # PriorityQueue
    def put(self, item, block=True, timeout=None):
        if item in self.queue:  # O(n)
            return
        else:
            super().put(item, block, timeout)  # O(logn)

    def __bool__(self):
        """while Queue:"""
        return bool(self.queue)

    def __contains__(self, item):
        """pos in Queue"""
        return item in self.queue

    def __len__(self):
        """len(Queue)"""
        return len(self.queue)

    def __getitem__(self, idx):
        """Queue[i]"""
        return self.queue[idx]


class GridMap:
    def __init__(
            self,
            img_path: str,
            thresh: int,
            high: int,
            width: int,
    ):
        """
        Parameters
        ----------
        img_path : str
        thresh : int
        high : int
        width : int
        """
        self.__map_path = 'map.png'
        self.__path_path = 'path.png'

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # H,W,C
        thresh, map_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        map_img = cv2.resize(map_img, (width, high))
        cv2.imwrite(self.__map_path, map_img)

        self.map_array = np.array(map_img)
        """ndarray, H*W, 0"""
        self.high = high
        """ndarray"""
        self.width = width
        """ndarray"""

    def show_path(self, path_list, *, save=False):
        """
        Parameters
        ----------
        path_list : list[Node]
        save : bool, optional
        """

        if not path_list:
            print("\nPassing in an empty list, unable to draw\n")
            return
        if not hasattr(path_list[0], "x") or not hasattr(path_list[0], "y"):
            print("\nThere is no coordinate x or coordinate y attribute in the path node and cannot be drawn.\n")
            return

        x, y = [], []
        for p in path_list:
            x.append(p.x)
            y.append(p.y)

        fig, ax = plt.subplots()
        map_ = cv2.imread(self.__map_path)
        map_ = cv2.resize(map_, (self.width, self.high))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # R G B
        # img = img[:, :, ::-1] # R G B
        map_ = map_[::-1]
        ax.axis('off')
        ax.imshow(map_, extent=[0, self.width, 0, self.high])  # extent[x_min, x_max, y_min, y_max]
        ax.plot(x, y, c='r', label='path', linewidth=2)
        ax.scatter(x[0], y[0], c='c', marker='o', label='start', s=40, linewidth=2)
        ax.scatter(x[-1], y[-1], c='c', marker='x', label='end', s=40, linewidth=2)
        ax.invert_yaxis()
        ax.legend().set_draggable(True)
        plt.show()
        if save:
            plt.savefig(self.__path_path)


def tic():
    if 'global_tic_time' not in globals():
        global global_tic_time
        global_tic_time = []
    global_tic_time.append(time.time())


def toc(name='', *, digit=6):
    if 'global_tic_time' not in globals() or not global_tic_time:
        print('tic not set')
        return
    name = name + ' '
    print('%sElapsed time is %f seconds.\n' % (name, round(time.time() - global_tic_time.pop(), digit)))


def limit_angle(x, mode=1):
    """
    mode1 : (-inf, inf) -> (-π, π]
    mode2 : (-inf, inf) -> [0, 2π)
    """
    x = x - x // (2 * math.pi) * 2 * math.pi  # any -> [0, 2π)
    if mode == 1 and x > math.pi:
        return x - 2 * math.pi  # [0, 2π) -> (-π, π]
    return x


Queue_Type = 0
"""
# The structure of PriorityQueue used by OpenList
## 0 -> SetQueue
## 1 -> ListQueue
## 2 -> PriorityQueuePro
List/Set can update the parent and cost of nodes in OpenList, and the path found is better.
PriorityQueuePro is the fastest, but cannot update information and has a poor path
List is the slowest, Set is close to PriorityQueuePro or even faster
"""

IMAGE_PATH = 'image1.jpg'
THRESH = 172
HEIGHT = 350
WIDTH = 600

MAP = GridMap(IMAGE_PATH, THRESH, HEIGHT, WIDTH)

START = (0, 0)
END = (40, 100)

""" ---------------------------- Greedy Best First Search ---------------------------- """

if Queue_Type == 0:
    NodeQueue = SetQueue
elif Queue_Type == 1:
    NodeQueue = ListQueue
else:
    NodeQueue = PriorityQueuePro


class GBFS:
    """GBFS"""

    def __init__(
            self,
            start_pos=START,
            end_pos=END,
            map_array=MAP.map_array,
            move_step=3,
            move_direction=8,
    ):
        """
        GBFS

        Parameters
        ----------
        start_pos : tuple/list
        end_pos : tuple/list
        map_array : ndarray
        move_step : int
        move_direction : int (8 or 4)
        """
        self.map_array = map_array  # H * W

        self.width = self.map_array.shape[1]
        self.high = self.map_array.shape[0]

        self.start = Node(*start_pos)
        self.end = Node(*end_pos)

        # Error Check
        if not self._in_map(self.start) or not self._in_map(self.end):
            raise ValueError(f"x coordinate range 0~{self.width - 1}, y coordinate range 0~{self.height - 1}")
        if self._is_collided(self.start):
            raise ValueError(f"The starting point x coordinate or y coordinate is on the obstacle")
        if self._is_collided(self.end):
            raise ValueError(f"The end point x coordinate or y coordinate is on the obstacle")

        self.reset(move_step, move_direction)

    def reset(self, move_step=3, move_direction=8):
        self.__reset_flag = False
        self.move_step = move_step
        self.move_direction = move_direction
        self.close_set = set()
        self.open_queue = NodeQueue()
        self.path_list = []

    def search(self):
        return self.__call__()

    def _in_map(self, node: Node):
        return (0 <= node.x < self.width) and (0 <= node.y < self.high)

    def _is_collided(self, node: Node):
        return self.map_array[node.y, node.x] == 0

    def _move(self):
        @lru_cache(maxsize=3)
        def _move(move_step: int, move_direction: int):
            move = (
                [0, move_step],
                [0, -move_step],
                [-move_step, 0],
                [move_step, 0],
                [move_step, move_step],
                [move_step, -move_step],
                [-move_step, move_step],
                [-move_step, -move_step],
            )
            return move[0:move_direction]

        return _move(self.move_step, self.move_direction)

    def _update_open_list(self, curr: Node):
        """open_list"""
        for add in self._move():
            next_ = curr + add

            if not self._in_map(next_):
                continue

            if self._is_collided(next_):
                continue
            # CloseList
            if next_ in self.close_set:
                continue

            h = next_ - self.end
            next_.cost = h  # G = 0

            # open-list
            self.open_queue.put(next_)

            if h < 20:
                self.move_step = 1

    def __call__(self):
        """GBFS"""
        assert not self.__reset_flag, "call reset"
        print("searching...\n")

        self.open_queue.put(self.start)

        tic()
        while not self.open_queue.empty():
            # OpenList
            curr = self.open_queue.get()
            # OpenList
            self._update_open_list(curr)
            # CloseList
            self.close_set.add(curr)

            if curr == self.end:
                break
        print("Path search completed\n")
        toc()

        while curr.parent is not None:
            self.path_list.append(curr)
            curr = curr.parent
        self.path_list.reverse()

        self.__reset_flag = True

        return self.path_list


# debug
if __name__ == '__main__':
    p = GBFS()()
    MAP.show_path(p)
