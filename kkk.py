class FibroNumbers:
    def __init__(self, numbers):
        self.numbers = numbers

    def __iter__(self):
        self.i = 0
        self.a = 1
        self.b = 0
        return self

    def __next__(self):
        if self.numbers > self.i:
            self.i += 1
            self.a, self.b = self.b, self.a + self.b
            return self.i, self.a
        raise StopIteration


def main():
    for i, j in FibroNumbers(9):
        print(f'{i}) {j}')


if __name__ == '__main__':
    main()
