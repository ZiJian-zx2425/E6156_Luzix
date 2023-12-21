import { Injectable } from '@angular/core';
import {HttpClient, HttpParams} from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PatientsRecordsService {

  private apiUrl = 'https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/patients-records';
  private doctorsPatientsUrl = 'https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/doctors-patients';
  private searchbypatientUrl = 'https://crxrcf7ds8.execute-api.us-east-1.amazonaws.com/test/search-by-patient';

  constructor(private http: HttpClient) {}

  getPatients(doctorId: string): Observable<string[]> {
    return this.http.get<string[]>(`${this.doctorsPatientsUrl}?doctor_id=${doctorId}`);
  }

  getPatientRecords(patientNames: string[]): Observable<any[]> {
    // Construct query parameters from the patient names array
    let params = new HttpParams();
    for (let name of patientNames) {
      params = params.append('patient_names[]', name);
    }
    return this.http.get<any[]>(this.apiUrl, { params });
  }

  searchByPatientName(name: string): Observable<any[]> {
    const params = new HttpParams().set('name', name);
    return this.http.get<any[]>(this.searchbypatientUrl, { params });
  }


}
