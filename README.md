# dbs2dbz
Web-app for converting Dragon Ball Super animation into that of Dragon Ball Z via CycleGAN.  
• Created novel dataset: [dbs2dbz](https://www.kaggle.com/petruso/dbs2dbz) (~15k images)  
• Implemented CycleGAN model (Jun-Yan Zhu et al. 2017)  
• Served the model as a web-app via Flask and Streamlit  
• Deployed on Heroku (dbs2dbz.herokuapp.com)  
  
Results after training for 10 epochs at batch size 2:

![10_2_50gn](https://user-images.githubusercontent.com/72981484/145722972-5a0f2fc4-536d-4df3-9153-8d3def95850e.png)

![download (1)](https://user-images.githubusercontent.com/72981484/145722951-263f826d-770e-4743-9828-a986912dc7dc.png)

![10_2_50tr](https://user-images.githubusercontent.com/72981484/145722981-332b1c5a-ce24-4f96-b1d2-69e60f4ce5b7.png)
