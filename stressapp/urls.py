from django.urls import path
from . import views

urlpatterns = [
    # Public pages
    path('', views.index_view, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home_view, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('profile/', views.profile_view, name='profile'),
    path('aboutus/', views.aboutus_views, name="aboutus"),
    path('result/', views.views_result, name='result'),

    # Admin pages
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_home/', views.admin_home, name='admin_home'),
    path('admin_logout/', views.admin_logout, name='logout_admin'),
    path('users/', views.admin_manage_users, name='admin_manage_users'),
    path('delete-user/<int:user_id>/', views.admin_delete_user, name='admin_delete_user'),
    path('stress-reports/', views.admin_stress_list, name='admin_stress_list'),
    path('model-accuracy/', views.admin_model_accuracy, name='admin_model_accuracy'),  # ← newly added
]
