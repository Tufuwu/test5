# Generated by Django 3.1.2 on 2020-11-23 21:01

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_peeringdb", "0013_add_phone_help_text"),
    ]

    operations = [
        migrations.AddField(
            model_name="facility",
            name="floor",
            field=models.CharField(blank=True, max_length=255, verbose_name="Floor"),
        ),
        migrations.AddField(
            model_name="facility",
            name="suite",
            field=models.CharField(blank=True, max_length=255, verbose_name="Suite"),
        ),
        migrations.AddField(
            model_name="organization",
            name="floor",
            field=models.CharField(blank=True, max_length=255, verbose_name="Floor"),
        ),
        migrations.AddField(
            model_name="organization",
            name="suite",
            field=models.CharField(blank=True, max_length=255, verbose_name="Suite"),
        ),
    ]
