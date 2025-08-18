
docker login -u cn-north-4@HST3W9J0QDGQ1AUURLAA -p 682b647fb2da67372e5c17da87297edf6b7f56d0c9ac7563dc3daeaadbe386e9 swr.cn-north-4.myhuaweicloud.com 

sudo docker tag botcalltest_zr:latest swr.cn-north-4.myhuaweicloud.com/zhangrui/botcalltest_zr:latest
sudo docker push swr.cn-north-4.myhuaweicloud.com/zhangrui/botcalltest_zr:latest
