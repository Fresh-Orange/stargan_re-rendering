import torch
from torchvision import transforms as T
from PIL import Image

crop_size = 256
image_size = 128
transform = []
transform.append(T.CenterCrop(crop_size))
transform.append(T.Resize(image_size))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

model = torch.load("./model/facenet.ckpt")
model.eval()

posi = 379
nega = 0

angle = [-90, -45, 0, 45, 90]
for i in range(len(angle)):
    for k in range(i+1, len(angle)):
        image_a = Image.open("../AI_DATA/multi_pie_id_crop/{}/multipie_az{}el0_{}.png".format(posi, angle[i], posi))
        image_p = Image.open("../AI_DATA/multi_pie_id_crop/{}/multipie_az{}el0_{}.png".format(posi, angle[k], posi))
        image_n = Image.open("../AI_DATA/multi_pie_id_crop/{}/multipie_az{}el0_{}.png".format(nega, angle[i], nega))

        image_a = transform(image_a)
        image_a = image_a.unsqueeze(0).to("cuda")
        image_p = transform(image_p)
        image_p = image_p.unsqueeze(0).to("cuda")
        image_n = transform(image_n)
        image_n = image_n.unsqueeze(0).to("cuda")

        feature_a = model(image_a)
        feature_p = model(image_p)
        feature_n = model(image_n)

        # print(feature_a[0][:10])
        # print(feature_p[0][:10])
        # print(feature_n[0][:10])

        print("-----------  i = {}, k = {} -----------".format(i, k))

        print(torch.dist(feature_a, feature_p, 2))

        print(torch.dist(feature_a, feature_n, 2))

        print(torch.dist(feature_p, feature_n, 2))

