import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary')

# test score
# scoreSeg = loaded_model.evaluate_generator(test_generator, 400)
# print(scoreSeg)

img = load_img('test/yes/yes1.jpg', target_size=(224, 224))
img = img_to_array(img).astype('double')


explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img, loaded_model.predict, top_labels=1, num_samples=1000, hide_color=0)
print(explanation)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(array_to_img(temp), mask))
plt.show()
plt.savefig('explanation.png')
#Select the same class explained on the figures above.
ind =  explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()
