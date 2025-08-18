docker login -u cn-southwest-2@HST3WU1ZLQ6979GWGOQ5 -p 49f9a1afe6a0b10590dcc8ff2d46f1373830bc50928b69ae318ad45ca290d06d swr.cn-southwest-2.myhuaweicloud.com
sudo docker pull swr.cn-southwest-2.myhuaweicloud.com/miaoliuyang/hwbotcall:1.0


docker login -u cn-north-4@HST3W9J0QDGQ1AUURLAA -p 682b647fb2da67372e5c17da87297edf6b7f56d0c9ac7563dc3daeaadbe386e9 swr.cn-north-4.myhuaweicloud.com 

sudo docker tag botcalltest_zr:latest swr.cn-north-4.myhuaweicloud.com/zhangrui/botcalltest_zr:latest
sudo docker push swr.cn-north-4.myhuaweicloud.com/zhangrui/botcalltest_zr:latest
