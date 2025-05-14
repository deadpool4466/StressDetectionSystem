from django.db import models
from django.contrib.auth.models import User

# ProfileDetails model for storing user profile information
class ProfileDetails(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')])
    mobile = models.CharField(max_length=10)

    def __str__(self):
        return self.user.username  # Return the username of the user

# StressRecord model for storing stress-related data of the user
class StressRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to the user who submitted the data
    age = models.FloatField()
    heart_rate = models.FloatField()
    exercise_hours_per_week = models.FloatField()
    stress_level = models.FloatField()
    sedentary_hours_per_day = models.FloatField()
    bmi = models.FloatField()
    sleep_hours_per_day = models.FloatField()
    hrv = models.FloatField()
    eda = models.FloatField()
    skin_temperature = models.FloatField()
    respiration_rate = models.FloatField()
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')])
    diet = models.CharField(max_length=10, choices=[('Healthy', 'Healthy'), ('Unhealthy', 'Unhealthy')])
    prediction_result = models.CharField(max_length=10) 
    created_at = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"Stress record for {self.user.username} - {self.created_at}"
