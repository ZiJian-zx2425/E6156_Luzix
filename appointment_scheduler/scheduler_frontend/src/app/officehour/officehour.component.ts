import { Component } from '@angular/core';
import {OfficeHourService} from "../services/office-hour.service";
import {Observable} from "rxjs";
import {HttpErrorResponse} from "@angular/common/http";

@Component({
  selector: 'app-officehour',
  templateUrl: './officehour.component.html',
  styleUrls: ['./officehour.component.css']
})
export class OfficehourComponent {
  doctorName: string = '';
  officeHours: string = '';

  constructor(private officeHourService: OfficeHourService) {}

  onSubmit(): void {
    // Split the office hours string into an array
    const officeHoursArray = this.officeHours.split(',').map(time => time.trim());

    this.officeHourService.saveOfficeHours(this.doctorName, officeHoursArray).subscribe({
      next: (response) => {
        console.log('Office hours saved:', response);
        // Additional success handling
      },
      error: (error) => {
        console.error('Error saving office hours:', error);
        // Additional error handling
      }
    });
  }

}
