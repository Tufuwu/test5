# Generated by Django 2.2 on 2019-07-29 21:22

from __future__ import annotations

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("app_unique_together", "0001_initial")]

    operations = [
        migrations.AlterUniqueTogether(
            name="a", unique_together={("int_field", "char_field")}
        )
    ]
