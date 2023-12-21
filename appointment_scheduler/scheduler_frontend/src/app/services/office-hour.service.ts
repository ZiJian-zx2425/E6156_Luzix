// office-hour.service.ts
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import {HttpClient, HttpClientModule, HttpErrorResponse} from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class OfficeHourService {
  private officeHoursUrl = 'https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/officehours'; // URL to your Flask API
  constructor(private http: HttpClient) {}

  saveOfficeHours(doctorName: string, slots: string[]): Observable<any> {
    return this.http.post(this.officeHoursUrl, { doctor_name: doctorName, slots: slots });
  }

  getAllOfficeHours(): Observable<any> {
    return this.http.get(this.officeHoursUrl);
  }


}
