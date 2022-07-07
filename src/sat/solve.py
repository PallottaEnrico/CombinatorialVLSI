import time
from itertools import combinations

from z3 import And, Or, Bool, sat, Not, Solver
import numpy as np


# from sat_minelli import positive_range, indomain


class SATsolver:

    def __init__(self, timeout=300, rotation=False):
        self.timeout = timeout
        self.rotation = rotation

    def __init__(self, data, rotation, output_dir, timeout):
        self.data = data
        self.rotation = rotation
        if output_dir == "":
            output_dir = "../data/output_sat/"
        self.output_dir = output_dir
        self.timeout = timeout

    def solve(self):
        solutions = []
        for d in self.data:
            solution = self.solve_instance(d)
            ins_num = d[0]
            # write_solution(ins_num, solution[0], solution[1])
            print(solution[1])
            solutions.append((ins_num, solution[0], solution[1]))
        return solutions

    def solve_instance(self, instance):
        _, self.max_width, self.circuits = instance
        self.circuits_num = len(self.circuits)

        self.w, self.h = ([i for i, _ in self.circuits], [j for _, j in self.circuits])

        lower_bound = int(round(sum([self.h[i] * self.w[i] for i in range(self.circuits_num)]) / self.max_width))
        upper_bound = sum(self.h) - min(self.h)

        if self.rotation:
            temp_w = self.w + self.h
            self.h = self.h + self.w
            self.w = temp_w
            self.circuits_num *= 2

        start_time = time.time()
        try_timeout = self.timeout
        for plate_height in range(lower_bound, upper_bound + 1):
            self.sol = Solver()
            self.sol.set(timeout=self.timeout * 1000)
            px, py = self.set_constraints(plate_height)

            solve_time = time.time()
            if self.sol.check() == sat:
                circuits_pos = self.evaluate(plate_height, px, py)
                return ((self.max_width, plate_height), circuits_pos), (time.time() - solve_time)
            else:
                try_timeout = round((self.timeout - (time.time() - start_time)))
                if try_timeout < 0:
                    return None, 0
        return None, 0

    def all_true(self, bool_vars):
        return And(bool_vars)

    def at_most_one(self, bool_vars):
        return [Not(And(pair[0], pair[1])) for pair in combinations(bool_vars, 2)]

    def exactly_one(self, bool_vars):
        self.sol.add(self.at_most_one(bool_vars))
        self.sol.add(Or(bool_vars))

    def set_constraints(self, plate_height):
        print(self.max_width)
        # Variables
        px = [[Bool(f"px_{i + 1}_{e}") for e in range(0, self.max_width - self.w[i] + 1)] for i in
              range(self.circuits_num)]
        py = [[Bool(f"py_{i + 1}_{f}") for f in range(0, plate_height - self.h[i] + 1)] for i in
              range(self.circuits_num)]

        lr = [[Bool(f"lr_{i + 1}_{j + 1}") for j in range(self.circuits_num)] for i in
              range(self.circuits_num)]
        ud = [[Bool(f"ud_{i + 1}_{j + 1}") for j in range(self.circuits_num)] for i in
              range(self.circuits_num)]

        print("Px \n", px)
        print("Py \n", py)
        print("Lr \n", lr)
        print("Ud \n", ud)

        if not self.rotation:
            # Place a circuit one time
            for x in px:
                self.sol.add(Or([i for i in x]))

            for y in py:
                self.sol.add(Or([i for i in y]))

            #
            # Order constraint
            for i in range(self.circuits_num):
                self.sol.add(px[i][-1])
                self.sol.add(py[i][-1])
                for j in range(self.max_width - self.w[i]):
                    self.sol.add(Or([Not(px[i][j]), px[i][j + 1]]))
                for j in range(plate_height - self.h[i]):
                    self.sol.add(Or([Not(py[i][j]), py[i][j + 1]]))

                #
            # Non overlapping no rotations
            for i in range(0, self.circuits_num):
                for j in range(0, self.circuits_num):
                    if i < j:
                        self.add_no_overlap(px, py, lr, ud, self.w, self.h, i, j, self.max_width, plate_height)
        else:
            rotation_index = int(self.circuits_num / 2)
            for i in range(rotation_index):
                # Considering a block and its rotation exactly one can have the max x/y to true.
                self.exactly_one([px[i][-1], px[i + rotation_index][-1]])
                self.exactly_one([py[i][-1], py[i + rotation_index][-1]])
                # self.sol.add()
                for j in range(self.max_width - self.w[i]):
                    self.sol.add(Or([Not(px[i][j]), px[i][j + 1]]))
                # self.sol.add(Or([Not(px[i][-3]), px[i][-2], Not(px[i][-1])]))
                for j in range(plate_height - self.h[i]):
                    self.sol.add(Or([Not(py[i][j]), py[i][j + 1]]))
                # self.sol.add(Or([Not(py[i][-3]), py[i][-2], Not(py[i][-1])]))

                # Considering a block and its rotation, only one can have at least 1 position for x and y
                self.sol.add(Or(
                    And(Or(px[i]), Not(Or(px[i + rotation_index])),
                        Or(py[i]), Not(Or(py[i + rotation_index]))),
                    And(Not(Or(px[i])), Or(px[i + rotation_index]),
                        Not(Or(py[i])), Or(py[i + rotation_index]))))

            for i in range(rotation_index, self.circuits_num):
                for j in range(self.max_width - self.w[i]):
                    self.sol.add(Or([Not(px[i][j]), px[i][j + 1]]))
                # self.sol.add(Or([Not(py[i][-3]), py[i][-2], Not(py[i][-1])]))
                for j in range(plate_height - self.h[i]):
                    self.sol.add(Or([Not(py[i][j]), py[i][j + 1]]))
                # self.sol.add(Or([Not(px[i][-3]), px[i][-2], Not(px[i][-1])]))

            # no overlap with rotations
            for i in range(0, self.circuits_num):
                for j in range(0, self.circuits_num):
                    if i < j and j != i + rotation_index:
                        self.add_no_overlap(px, py, lr, ud, self.w, self.h, i, j, self.max_width, plate_height)
                    # if i < j and j != i + rotation_index:
                    #     if i < rotation_index and j < rotation_index:
                    #         self.add_no_overlap(px, py, lr, ud, self.w, self.h, i, j, self.max_width, plate_height)
                    #     elif i < rotation_index and j >= rotation_index:
                    #         self.add_no_overlap(px, py, lr, ud, self.w, self.h, i, j, self.max_width, plate_height)
                    #         # self.add_no_overlap_norot_rot(px, py, lr, ud, self.w, self.h, i, j, self.max_width,
                    #         #                               plate_height)
                    #     elif i >= rotation_index and j >= rotation_index:
                    #         self.add_no_overlap(px, py, lr, ud, self.w, self.h, i, j, self.max_width, plate_height)
                    #         # self.add_no_overlap(py, px, lr, ud, self.h, self.w, i, j, self.max_width, plate_height)

        return px, py

    def add_no_overlap(self, px, py, lr, ud, w, h, i, j, max_w, max_h):
        if len(px[j]) - 1 >= w[i] - 1:
            self.sol.add(Or(Not(lr[i][j]), Not(px[j][w[i] - 1])))
        else:
            # print("Else 1:", Or(Not(lr[i][j]), Not(px[j][-1])))
            self.sol.add(Or(Not(lr[i][j]), Not(px[j][-1])))
        if len(px[i]) - 1 >= w[j] - 1:
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][w[j] - 1])))
        else:
            # print("Else 2:", Or(Not(lr[j][i]), Not(px[i][-1])))
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][-1])))
        if len(py[j]) - 1 >= h[i] - 1:
            self.sol.add(Or(Not(ud[i][j]), Not(py[j][h[i] - 1])))
        else:
            # print("Else 3:", Or(Not(ud[i][j]), Not(py[j][-1])))
            self.sol.add(Or(Not(ud[i][j]), Not(py[j][-1])))
        if len(py[i]) - 1 >= h[j] - 1:
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][h[j] - 1])))
        else:
            # print("Else 4:", Or(Not(ud[j][i]), Not(py[i][-1])))
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][-1])))

        self.sol.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

        for e in range(max_w - w[i]):
            if len(px[j]) - 1 >= e + w[i]:
                self.sol.add(Or(Not(lr[i][j]), px[i][e], Not(px[j][e + w[i]])))
            if len(px[i]) - 1 >= e + w[j]:
                self.sol.add(Or(Not(lr[j][i]), px[j][e], Not(px[i][e + w[j]])))
        for f in range(max_h - h[j]):
            if len(py[j]) - 1 >= f + h[i]:
                self.sol.add(Or(Not(ud[i][j]), py[i][f], Not(py[j][f + h[i]])))
            if len(py[i]) - 1 >= f + h[j]:
                self.sol.add(Or(Not(ud[j][i]), py[j][f], Not(py[i][f + h[j]])))

    def add_no_overlap_nuovo(self, px, py, lr, ud, w, h, i, j, max_w, max_h):
        if len(px[j]) >= w[i]:
            self.sol.add(Or(Not(lr[i][j]), Not(px[j][w[i] - 1])))
        else:
            self.sol.add(Or(Not(lr[i][j]), Not(px[j][-1])))
        if len(px[i]) - 1 >= w[j] - 1:
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][w[j] - 1])))
        else:
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][-1])))
        if len(py[j]) - 1 >= h[i] - 1:
            self.sol.add(Or(Not(ud[i][j]), Not(py[j][h[i] - 1])))
        else:
            self.sol.add(Or(Not(ud[i][j]), Not(py[j][-1])))
        if len(py[i]) - 1 >= h[j] - 1:
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][h[j] - 1])))
        else:
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][-1])))

        self.sol.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

        for e in range(max_w - w[i]):
            if len(px[j]) - 1 >= e + w[i]:
                self.sol.add(Or(Not(lr[i][j]), px[i][e], Not(px[j][e + w[i]])))
            if len(px[i]) - 1 >= e + w[j]:
                self.sol.add(Or(Not(lr[j][i]), px[j][e], Not(px[i][e + w[j]])))
        for f in range(max_h - h[j]):
            if len(py[j]) - 1 >= f + h[i]:
                self.sol.add(Or(Not(ud[i][j]), py[i][f], Not(py[j][f + h[i]])))
            if len(py[i]) - 1 >= f + h[j]:
                self.sol.add(Or(Not(ud[j][i]), py[j][f], Not(py[i][f + h[j]])))

    def add_no_overlap_norot_rot(self, px, py, lr, ud, w, h, i, j, max_w, max_h):
        if len(py[j]) - 1 >= w[i] - 1:
            self.sol.add(Or(Not(lr[i][j]), Not(py[j][w[i] - 1])))
        else:
            self.sol.add(Or(Not(lr[i][j]), Not(py[j][-1])))
        if len(px[i]) - 1 >= h[j] - 1:
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][h[j] - 1])))
        else:
            self.sol.add(Or(Not(lr[j][i]), Not(px[i][-1])))
        if len(px[j]) - 1 >= h[i] - 1:
            self.sol.add(Or(Not(ud[i][j]), Not(px[j][h[i] - 1])))
        else:
            self.sol.add(Or(Not(ud[i][j]), Not(px[j][-1])))
        if len(py[i]) - 1 >= w[j] - 1:
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][w[j] - 1])))
        else:
            self.sol.add(Or(Not(ud[j][i]), Not(py[i][-1])))

        self.sol.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

        for e in range(max_w - w[i]):
            if len(py[j]) - 1 >= e + w[i]:
                self.sol.add(Or(Not(lr[i][j]), px[i][e], Not(py[j][e + w[i]])))
            if len(px[i]) - 1 >= e + h[j]:
                self.sol.add(Or(Not(lr[j][i]), py[j][e], Not(px[i][e + h[j]])))
        for f in range(max_h - w[j]):
            if len(px[j]) - 1 >= f + h[i]:
                self.sol.add(Or(Not(ud[i][j]), py[i][f], Not(px[j][f + h[i]])))
            if len(py[i]) - 1 >= f + w[j]:
                self.sol.add(Or(Not(ud[j][i]), px[j][f], Not(py[i][f + w[j]])))

    def add_no_overlap_rot_rot(self, px, py, lr, ud, w, h, i, j, max_w, max_h):
        if len(py[j]) - 1 >= h[i] - 1:
            self.sol.add(Or(Not(lr[i][j]), Not(py[j][h[i] - 1])))
        else:
            self.sol.add(Or(Not(lr[i][j]), Not(py[j][-1])))
        if len(py[i]) - 1 >= h[j] - 1:
            self.sol.add(Or(Not(lr[j][i]), Not(py[i][h[j] - 1])))
        else:
            self.sol.add(Or(Not(lr[j][i]), Not(py[i][-1])))
        if len(px[j]) - 1 >= w[i] - 1:
            self.sol.add(Or(Not(ud[i][j]), Not(px[j][w[i] - 1])))
        else:
            self.sol.add(Or(Not(ud[i][j]), Not(px[j][-1])))
        if len(px[i]) - 1 >= w[j] - 1:
            self.sol.add(Or(Not(ud[j][i]), Not(px[i][w[j] - 1])))
        else:
            self.sol.add(Or(Not(ud[j][i]), Not(px[i][-1])))

        self.sol.add(Or(lr[i][j], lr[j][i], ud[i][j], ud[j][i]))

        for e in range(max_w - h[i]):
            if len(py[j]) - 1 >= e + h[i]:
                self.sol.add(Or(Not(lr[i][j]), py[i][e], Not(py[j][e + h[i]])))
            if len(py[i]) - 1 >= e + h[j]:
                self.sol.add(Or(Not(lr[j][i]), py[j][e], Not(py[i][e + h[j]])))
        for f in range(max_h - w[j]):
            if len(px[j]) - 1 >= f + w[i]:
                self.sol.add(Or(Not(ud[i][j]), px[i][f], Not(px[j][f + w[i]])))
            if len(px[i]) - 1 >= f + w[j]:
                self.sol.add(Or(Not(ud[j][i]), px[j][f], Not(px[i][f + w[j]])))

    def evaluate(self, plate_height, px, py):
        m = self.sol.model()

        circuits_pos = []
        X = []
        Y = []

        if self.rotation:
            R = np.zeros(int(self.circuits_num / 2))

        for i in range(self.circuits_num):
            for e in range(len(px[i])):
                print(px[i][e], m.evaluate(px[i][e]))

        for i in range(self.circuits_num):
            for e in range(len(px[i])):
                # print(px[i][e], m.evaluate(px[i][e]))
                if str(m.evaluate(px[i][e])) == 'True':
                    X.append(e);
                    if self.rotation:
                        if i >= int(self.circuits_num / 2):
                            R[i - int(self.circuits_num / 2)] = 1
                        else:
                            R[i] = 0
                    break
            for f in range(len(py[i])):
                # print(py[i][f], m.evaluate(py[i][f]))
                if str(m.evaluate(py[i][f])) == 'True':
                    Y.append(f);
                    break
        print(X)
        print(Y)
        if self.rotation:
            print(R)

        if not self.rotation:
            for i, (x, y) in enumerate(zip(X, Y)):
                circuits_pos.append((self.w[i], self.h[i], x, y))
        else:
            for i, (x, y, r) in enumerate(zip(X, Y, R)):
                if r == 1:
                    circuits_pos.append((self.h[i], self.w[i], x, y))
                else:
                    circuits_pos.append((self.w[i], self.h[i], x, y))
        # for k in range(self.circuits_num):
        #     found = False
        #     for x in range(self.max_width):
        #         if found:
        #             break
        #         for y in range(plate_height):
        #             if not found and m.evaluate(plate[x][y][k]):
        #                 if not self.rotation:
        #                     circuits_pos.append((self.w[k], self.h[k], x, y))
        #                 else:
        #                     if m.evaluate(rotations[k]):
        #                         circuits_pos.append((self.h[k], self.w[k], x, y))
        #                     else:
        #                         circuits_pos.append((self.w[k], self.h[k], x, y))
        #                 found = True
        #             elif found:
        #                 break

        return circuits_pos
