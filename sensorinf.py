from sklearn import svm

is_first = True

test_data_x = [[800], [199], [111], [430]]

train_data_x = [[700], [300], [800], [900]]
train_data_y = ['good', 'good', 'bad', 'bad']

model = svm.SVC(kernel="linear")


def train_model():
    global is_first
    is_first = False
    model.fit(train_data_x, train_data_y)
    pass


def predict_model(input_data):
    if is_first:
        train_model()
    prediction = model.predict(input_data)
    print(prediction)
    return str(prediction[0])
