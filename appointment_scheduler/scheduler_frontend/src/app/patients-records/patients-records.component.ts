import {Component, OnInit} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {PatientsRecordsService} from "../patients-records.service";
import {AuthService} from "../services/auth/auth.service";

@Component({
  selector: 'app-patients-records',
  templateUrl: './patients-records.component.html',
  styleUrls: ['./patients-records.component.css']
})
export class PatientsRecordsComponent implements OnInit {
  patientRecords: any[] = [];
  patients: any[] = [];
  doctorId: string = ''; // This should be set based on the logged-in doctor's google_id

  constructor(private authService: AuthService,
              private patientsRecordsService: PatientsRecordsService) {}

  ngOnInit(): void {
    this.doctorId = this.authService.getUserGoogleId();
    console.log(this.doctorId) // 104405107080836112407

    this.patientsRecordsService.getPatients(this.doctorId).subscribe(data => {
      this.patients = data;
      console.log('Patients related to this doctor:', this.patients);

      this.fetchPatientRecords();
    });
  }

  fetchPatientRecords(): void {
    // Assuming getPatientRecords method takes an array of patient names
    // and fetches their records
    if (this.patients.length > 0) {
      this.patientsRecordsService.getPatientRecords(this.patients).subscribe(records => {
        this.patientRecords = records;
        console.log('Patient Records:', this.patientRecords);
      }, error => {
        console.error('Error fetching patient records:', error);
      });
    }
  }
}
