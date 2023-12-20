// postedofficehours.component.ts

import { Component, OnInit } from '@angular/core';
import {OfficeHourService} from "../services/office-hour.service";

interface DoctorOfficeHours {
  [doctorName: string]: string[];
}

@Component({
  selector: 'app-postedofficehours',
  templateUrl: './postedofficehours.component.html',
  styleUrls: ['./postedofficehours.component.css']
})
export class PostedOfficeHoursComponent implements OnInit {
  allOfficeHours:  DoctorOfficeHours = {};

  constructor(private officeHourService: OfficeHourService) {}

  ngOnInit(): void {
    this.officeHourService.getAllOfficeHours().subscribe({
      next: (data) => {
        this.allOfficeHours = data;
        // console.error('allOfficeHours data fetched successfully');
      },
      error: (error) => {
        console.error('Error fetching office hours:', error);
      }
    });
  }
}
