import { Component } from '@angular/core';
import {PatientsRecordsService} from "../patients-records.service";

@Component({
  selector: 'app-search-by-patient',
  templateUrl: './search-by-patient.component.html',
  styleUrls: ['./search-by-patient.component.css']
})
export class SearchByPatientComponent {
  patientRecords: any[] = [];
  searchQuery: string = '';

  constructor(private patientsRecordsService: PatientsRecordsService) {}

  searchPatients(): void {
    this.patientsRecordsService.searchByPatientName(this.searchQuery).subscribe(data => {
      this.patientRecords = data;
    });
  }
}
