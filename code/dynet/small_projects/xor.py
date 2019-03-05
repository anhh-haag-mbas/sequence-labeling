import dynet as dy
import numpy as np
import ipdb

def test(HIDDEN_SIZE):
    ITERATIONS  = 2000

    m = dy.Model()
    trainer = dy.SimpleSGDTrainer(m)

    W = m.add_parameters((HIDDEN_SIZE, 2))
    b = m.add_parameters(HIDDEN_SIZE)
    V = m.add_parameters((1, HIDDEN_SIZE))
    a = m.add_parameters(1)

    x = dy.vecInput(2)
    y = dy.scalarInput(0)
    h = dy.tanh((W*x) + b)

    y_pred = dy.logistic((V*h) + a)
    loss = dy.binary_log_loss(y_pred, y)

    for i in range(ITERATIONS):
        mloss = 0.0
        x1s = [1, 1, 0, 0]
        x2s = [1, 0, 1, 0]
        outs  = [0, 1, 1, 0]

        for x1, x2, out in zip(x1s, x2s, outs):
            x.set((x1, x2))
            y.set(out)
            mloss += loss.scalar_value()
            loss.backward()
            trainer.update()

        mloss /= 4
    #print(f"loss: {mloss}")

    x.set([1, 0])
    z = -(-y_pred)
    print(z.scalar_value())

    dy.renew_cg()

    x = dy.vecInput(2)
    y = dy.scalarInput(0)
    h = dy.tanh((W*x) + b)

    y_pred = dy.logistic((V*h) + a)

    T = 1
    F = 0

    x.set([T,F])
    print("TF",y_pred.scalar_value())
    x.set([F,F])
    print("FF",y_pred.scalar_value())
    x.set([T,T])
    print("TT",y_pred.scalar_value())
    x.set([F,T])
    print("FT",y_pred.scalar_value())

for i in range(1, 8):
    print (f"i = {i}")
    test(i)
