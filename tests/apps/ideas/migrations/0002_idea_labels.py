# Generated by Django 2.2.24 on 2021-07-09 09:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("a4labels", "0001_initial"),
        ("a4test_ideas", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="idea",
            name="labels",
            field=models.ManyToManyField(
                related_name="a4test_ideas_idea_label", to="a4labels.Label"
            ),
        ),
    ]
