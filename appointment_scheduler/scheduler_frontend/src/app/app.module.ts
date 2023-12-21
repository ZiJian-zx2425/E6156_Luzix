import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';
import { AppRoutingModule } from './app-routing.module';
import { ScheduleComponent } from './schedule/schedule.component';
import {FormsModule} from "@angular/forms";
import { AppointmentListComponent } from './appointment-list/appointment-list.component';
import { AppointmentEditComponent } from './appointment-edit/appointment-edit.component';
import { NavbarComponent } from './navbar/navbar.component';
import { AuthSuccessComponent } from './auth-success/auth-success.component';
import { OfficehourComponent } from './officehour/officehour.component';
import {PostedOfficeHoursComponent} from "./postedofficehours/postedofficehours.component";
import { PatientsRecordsComponent } from './patients-records/patients-records.component';
import { SearchByPatientComponent } from './search-by-patient/search-by-patient.component';


@NgModule({
  declarations: [
    AppComponent,
    ScheduleComponent,
    AppointmentListComponent,
    AppointmentEditComponent,
    NavbarComponent,
    AuthSuccessComponent,
    OfficehourComponent,
    PostedOfficeHoursComponent,
    PatientsRecordsComponent,
    SearchByPatientComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
