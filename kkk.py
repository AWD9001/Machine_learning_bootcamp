"""class FunkcjaLiczbowa:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.start > self.stop:
                raise StopIteration
            else:
                self.start += 1
                return self.start - 1


def main():
    a = 1
    b = 10
    liczbowa = FunkcjaLiczbowa(a, b)
    for x in liczbowa:
        print(x)


if __name__ == '__main__':
    main()
"""


class DoListy:
    def __init__(self, to_read):
        self.to_read = to_read

    def __iter__(self):
        self.i = 0
        self.max = len(self.to_read)
        return self

    def __next__(self):
        if self.i >= self.max:
            raise StopIteration
        else:
            self.i += 1
            return self.to_read[self.i - 1]

    def __repr__(self):
        return 'repozytoria'


def main():
    lista = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    odczytaj = DoListy(lista)
    for x in odczytaj:
        print(x)
    print(odczytaj)


if __name__ == '__main__':
    main()