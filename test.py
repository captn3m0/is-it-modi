#JUST TO TEST
img=image.load_img('Data/Validation/120.jpg',target_size=(250,250))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
ans = model.predict(images)
print(ans)