import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)


logo = Image.open(r'images\logo.png')
st.image(logo, width=800)

st.title("Website Xử lý ảnh số")
st.caption("Thực hiện bởi: Tạ Nghĩa Nhân và Đặng Đăng Duy")
st.caption("Giảng viên hướng dẫn: ThS. Trần Tiến Đức")
st.caption("Lớp xử lý ảnh số nhóm 01: DIPR430685_23_2_02")

st.markdown(" Thành viên thực hiện ")
left, right = st.columns(2)
with left: 
   st.image(Image.open(r'images\TaNghiaNhan.jpg'), "Tạ Nghĩa Nhân", width=350)
with right:
     st.image(Image.open(r'images\DangDangDuy.jpg'), "Đặng Đăng Duy", width=350)


st.markdown(
    """
    ### Thông tin liên hệ   	
    - Email: 22110388@student.hcmute.edu.vn hoặc 22110295@student.hcmute.edu.vn    

    ### Video demo chức năng hoạt động của web:

    ### Nguồn tham khảo: https://discuss.streamlit.io (Của hai anh: Phan Hồng Sơn và Nguyễn Thành Đạt)
    """
)
    