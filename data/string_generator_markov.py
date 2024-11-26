import random
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from typing import List
import re


unit_operations = {
    "abs": {"inputs": (2, 2), "outputs": (2, 2), "tags": ["bin", "tin", "bout", "tout"]},
    "blwr": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "centr": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "comp": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "cond": {"inputs": (1, 1), "outputs": (2, 2), "tags": ["bout", "tout"]},
    "cycl": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "dist": {"inputs": (1, 2), "outputs": (2, 2), "tags": ["bout", "tout"]},
    "expand": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "extr": {"inputs": (2, 2), "outputs": (2, 2), "tags": []},
    "flash": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "gfil": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "hex": {"inputs": (1, 2), "outputs": (1, 2), "tags": ["he"]},
    "lfil": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "mix": {"inputs": (2, 2), "outputs": (1, 1), "tags": []},
    "r": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "raw": {"inputs": (0, 0), "outputs": (1, 1), "tags": []},
    "pp": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "prod": {"inputs": (1, 1), "outputs": (0, 0), "tags": []},
    "splt": {"inputs": (1, 1), "outputs": (2, 2), "tags": []},
    "v": {"inputs": (1, 1), "outputs": (1, 1), "tags": []},
    "rect": {"inputs": (1, 2), "outputs": (2, 2), "tags": []}
}


class State(Enum):
    FEED = 0
    REACTION = 1
    THERMAL_SEPARATION = 2
    CC_SEPARATION = 3
    FILTRATION = 4
    CENTRIFUGATION = 5
    NEXT = 6
    PURIFICATION = 7
    RECYCLE = 8
    PRODUCT = 9

@dataclass
class Unit:
    name: str
    filled_inputs: int = 1
    filled_outputs: int = 0

class sfilesGenerator:
    recycle_count = 1
    depth = 1
    def __init__(self, state = State.FEED):
        self.units: List[Unit] = []
        self.state = state

    def add_unit(self) -> Unit:
        unit_name = None
        if self.state == State.FEED:
            unit_name = "(raw)"
        elif self.state == State.REACTION:
            unit_name = "(r)"
        elif self.state == State.THERMAL_SEPARATION:
            if random.random() < 0.5:
                unit_name = "(dist)"
            else:
                unit_name = "(rect)"
        elif self.state == State.CC_SEPARATION:
            if random.random() < 0.5:
                unit_name = "(abs)"
            else:
                unit_name = "(extr)"
        elif self.state == State.FILTRATION:
            if random.random() < 0.5:
                unit_name = "(gfil)"
            else:
                unit_name = "(lfil)"
        elif self.state == State.CENTRIFUGATION:
            unit_name = "(centr)"
        else:
            raise ValueError(f"Invalid state: {self.state}")
        unit = Unit(unit_name)
        if unit.name == "(raw)":
            unit.filled_inputs = 0
        self.units.append(unit)
        return unit

    def add_new_inlet(self):
        sfilesGenerator.depth += 1
        gen = sfilesGenerator()
        units = gen.generate_string()
        if units[0].name == "(prod)":
            return
        units[0].name = "<&|" + units[0].name 
        rand = random.random()
        if rand < 0.2:
            rand_idx = random.randint(0, len(gen.units)) - 2
            rand_idx = rand_idx if rand_idx > 0 else 0
            units[rand_idx].name = units[rand_idx].name + "&"
            units[-1].name = units[-1].name + "|"
        else:
            units.pop()
            units[-1].name = units[-1].name + "&|"
        self.units.extend(units)

    def add_new_outlet(self):
        sfilesGenerator.depth += 1
        gen = sfilesGenerator(state=self.state)
        units = gen.generate_string()
        units[0].name = "[" + units[0].name
        units[-1].name = units[-1].name + "]"
        units[-1].filled_outputs += 1
        self.units.extend(units)

    def generate_string(self) -> str:
        if sfilesGenerator.depth >= 5:
            self.units.append(Unit("(prod)"))
            return self.units
        
        while True:
            if self.state == State.FEED:
                unit = self.add_unit()
                rand = random.random()
                if rand < 0.25:
                    self.state = State.REACTION
                elif rand < 0.65:
                    self.state = State.THERMAL_SEPARATION
                elif rand < 0.9:
                    self.state = State.CC_SEPARATION
                elif rand < 0.95:
                    self.state = State.FILTRATION
                else:
                    self.state = State.CENTRIFUGATION
            elif self.state.value > 0 and self.state.value < 6:
                unit = self.add_unit()
                #  outlet 2 이상이면 대괄호 추가하여 나가는 분기 표현
                name = unit.name
                name = re.sub(r'[^a-zA-Z]', '', name)  # 알파벳만 남기고 모두 제거
                if unit_operations[name]["outputs"][1] >= 2 and unit.filled_outputs < unit_operations[name]["outputs"][1]:
                    if unit_operations[name]["outputs"][0] >= 2:
                        unit.filled_outputs += 1
                        self.add_new_outlet()
                    else:
                        rand = random.random()
                        if rand < 0.2:
                            unit.filled_outputs += 1
                            self.add_new_outlet()
                if name == "dist" or "r":
                    idx = self.units.index(unit)
                    rand = random.random()
                    if rand < 0.5:
                        self.units.insert(idx + 2, Unit("(hex)"))
                        rand = random.random()
                        if rand < 0.5:
                            self.units.insert(idx + 3, Unit("(comp)"))


                self.state = State.NEXT
            elif self.state == State.NEXT:
                rand = random.random()
                if rand < 0.125:
                    self.state = State.REACTION
                elif rand < 0.325:
                    self.state = State.THERMAL_SEPARATION
                elif rand < 0.45:
                    self.state = State.CC_SEPARATION
                elif rand < 0.475:
                    self.state = State.FILTRATION
                elif rand < 0.5:
                    self.state = State.CENTRIFUGATION
                else:
                    self.state = State.PURIFICATION
            elif self.state == State.PURIFICATION:
                rand = random.random()
                if rand < 0.2:
                    self.state = State.RECYCLE
                else:
                    self.state = State.PRODUCT
            elif self.state == State.RECYCLE:
                # 재순환 로직 추가, <n n 활용
                if sfilesGenerator.recycle_count <= 3:   
                    if len(self.units) <= 1:
                        print("length is 1!")
                    else:
                        idx = -1
                        while True:
                            name = re.sub(r'[^a-zA-Z]', '', self.units[idx].name)
                            if name == "prod":
                                idx -= 1
                            else:
                                break
                        self.units[idx].name = self.units[idx].name + str(sfilesGenerator.recycle_count)
                        rand_idx = random.randint(0, len(self.units) + idx) - 2
                        rand_idx = rand_idx if rand_idx > 0 else 0
                        self.units[rand_idx].name = self.units[rand_idx].name + "<" + str(sfilesGenerator.recycle_count)
                        sfilesGenerator.recycle_count += 1
                self.state = State.PRODUCT
            elif self.state == State.PRODUCT:
                # TODO : product state 갔는데 inlet 남아있으면 <%| 로 raw 새로 추가
                for unit in self.units:
                    name = re.sub(r'[^a-zA-Z]', '', unit.name)  # 알파벳만 남기고 모두 제거
                    if unit_operations[name]["inputs"][1] >= 2 and unit_operations[name]["inputs"][1] > unit.filled_inputs:
                        if unit_operations[name]["inputs"][0] >= 2:
                            unit.filled_inputs += 1
                            self.add_new_inlet()
                        else:
                            rand = random.random()
                            if rand < 0.2:
                                unit.filled_inputs += 1
                                self.add_new_inlet()
                unit = Unit("(prod)")
                self.units.append(unit)
                break


        return self.units 


def generate_sfiles_strings(n: int, train_file: str = "train.txt", val_file: str = "validation.txt"):
    """
    n개의 SFILES 2.0 문자열을 생성하여 9:1 비율로 train.txt와 validation.txt로 저장
    
    Args:
        n: 생성할 문자열 총 개수
        train_file: 학습 데이터 파일 경로
        val_file: 검증 데이터 파일 경로
    """
    # 9:1 비율로 분할
    train_size = int(n * 0.9)
    val_size = n - train_size
    
    # 모든 문자열 생성
    strings = []
    for _ in range(n):
        # 매 반복마다 recycle_count와 depth 초기화
        sfilesGenerator.recycle_count = 1
        sfilesGenerator.depth = 1
        
        # 새로운 문자열 생성
        sfiles = sfilesGenerator().generate_string()
        string = "".join([unit.name for unit in sfiles])
        strings.append(string)
    
    # train.txt에 저장 (90%)
    with open(train_file, 'w', encoding='utf-8') as f:
        for string in strings[:train_size]:
            f.write(string + '\n')
            
    # validation.txt에 저장 (10%)
    with open(val_file, 'w', encoding='utf-8') as f:
        for string in strings[train_size:]:
            f.write(string + '\n')

# 사용 예시
generate_sfiles_strings(1000)  # 100개의 문자열 생성