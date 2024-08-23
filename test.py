class Test:
    def __init__(self, name):
        self.name = name
        

    def say_hello(self):
        
        print("Hello, " + self.name + "!")
        

t = Test("Joshen")
t.say_hello()